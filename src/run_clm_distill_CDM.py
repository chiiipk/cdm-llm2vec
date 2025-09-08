import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from transformers.testing_utils import CaptureLogger
from transformers.utils import is_sagemaker_mp_enabled
from transformers.optimization import get_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional
import deepspeed
import accelerate
from accelerate.utils import DummyOptim
import numpy as np
import json
import datasets
#import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import transformers
print(transformers.__version__)
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers import DataCollatorForLanguageModeling
from my_trainer import Trainer
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

import time
import copy
import editdistance

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.LlamaTokenizerFast: "▁",
    transformers.GPTNeoXTokenizerFast: "Ġ",
    transformers.GPT2Tokenizer: "Ġ",
    transformers.GPT2TokenizerFast: "Ġ",
    transformers.Qwen2Tokenizer: "Ġ",
    transformers.Qwen2TokenizerFast: "Ġ",
    transformers.BertTokenizer: "##",
    transformers.BertTokenizerFast: "##",
}

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def calculate_weight(logits):
    probabilities = torch.softmax(logits, dim=-1)
    epsilon = 1e-10
    probabilities = probabilities + epsilon
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    entropy_min = entropy.min()
    entropy_max = entropy.max()
    factor = torch.sigmoid((entropy - entropy_min) / (entropy_max - entropy_min) * 4 - 2)
    factor = torch.tensor((factor * 3 + 3), dtype=torch.int32).detach().cpu().tolist()
    return factor

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Student model checkpoint."},
    )
    teacher_path: Optional[str] = field(default=None, metadata={"help": "Comma-separated teacher path(s)."})
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    torch_dtype: Optional[str] = field(
        default=None, metadata={"choices": ["auto", "bfloat16", "float16", "float32"]}
    )

    freeze_emb: bool = field(default=False)
    freeze_layers: Optional[str] = field(default=None)

    kd_alpha: Optional[float] = field(default=0.9)
    kd_temperature: Optional[float] = field(default=0.8)

    # NEW: hỗ trợ student là encoder (MaskedLM/BERT)
    student_arch: Optional[str] = field(
        default="masked_lm",
        metadata={"help": "Student architecture: 'masked_lm' (BERT) hoặc 'causal_lm'."},
    )
    mlm_probability: Optional[float] = field(
        default=0.15,
        metadata={"help": "Tỷ lệ mask khi student_arch == 'masked_lm'."},
    )

    enable_edit_kd: Optional[bool] = field(default=False)
    enable_topk: Optional[bool] = field(default=False)
    topk: Optional[int] = field(default=100)
    simi_threadshold: Optional[float] = field(default=0.3)
    teacher_to_student_id_mapping: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides can't be used with --config_name or --model_name_or_path")


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    streaming: bool = field(default=False)
    shuffle_buffer_size: int = field(default=10000)
    block_size: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    keep_linebreaks: bool = field(default=True)
    padding_side: str = field(default=None)

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


# --------- Collator kết hợp cho CDM + MLM ----------
class CDMMLMCollator:
    """
    - Pad student bằng student tokenizer.
    - Tạo MLM labels cho student.
    - Pad toàn bộ teacher_* bằng teacher tokenizer.
    - Giữ teacher_labels0 và pad bằng IGNORE_INDEX.
    """
    def __init__(self, student_tokenizer, teacher_tokenizer, mlm_probability=0.15):
        self.stu_tok = student_tokenizer
        self.tea_tok = teacher_tokenizer
        if self.tea_tok.pad_token is None and self.tea_tok.eos_token is not None:
            self.tea_tok.pad_token = self.tea_tok.eos_token
        self.mlm = DataCollatorForLanguageModeling(
            tokenizer=student_tokenizer, mlm=True, mlm_probability=mlm_probability
        )

    def __call__(self, features):
        # Lưu teacher_* ra riêng
        tea_feats = []
        tea_labels_list = []
        for f in features:
            tea_feats.append({
                "input_ids": f["teacher_input_ids0"],
                "attention_mask": f["teacher_attention_mask0"],
            })
            if "teacher_labels0" in f:
                tea_labels_list.append(torch.tensor(f["teacher_labels0"], dtype=torch.long))

        # Tạo batch student (pad + MLM labels)
        # tokenizer.pad sẽ chỉ pad các khóa model_input_names; khóa khác trả về riêng
        stu_pad_batch = self.stu_tok.pad(features, padding=True, return_tensors="pt")

        # Chuẩn bị input cho MLM collator (chỉ giữ các khóa student)
        mlm_input = []
        keep_keys = {"input_ids", "attention_mask", "token_type_ids"}
        for f in features:
            mlm_input.append({k: v for k, v in f.items() if k in keep_keys})

        mlm_batch = self.mlm(mlm_input)
        stu_pad_batch["input_ids"] = mlm_batch["input_ids"]
        stu_pad_batch["attention_mask"] = mlm_batch["attention_mask"]
        stu_pad_batch["labels"] = mlm_batch["labels"]
        if "token_type_ids" in mlm_batch:
            stu_pad_batch["token_type_ids"] = mlm_batch["token_type_ids"]

        # Pad teacher side
        tea_batch = self.tea_tok.pad(tea_feats, padding=True, return_tensors="pt")
        stu_pad_batch["teacher_input_ids0"] = tea_batch["input_ids"]
        stu_pad_batch["teacher_attention_mask0"] = tea_batch["attention_mask"]

        if len(tea_labels_list) > 0:
            tea_labels = torch.nn.utils.rnn.pad_sequence(
                tea_labels_list, batch_first=True, padding_value=IGNORE_INDEX
            )
            stu_pad_batch["teacher_labels0"] = tea_labels

        return stu_pad_batch
# ---------------------------------------------------


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.disable_tqdm = True
    send_example_telemetry("run_clm", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    class CustomMetric:
        def __init__(self):
            self.sft_loss = 0
            self.distill_loss = 0
            self.unmasked_rate = 0
            self.count = 0

        def update(self, sft_loss, distill_loss, unmasked_rate=None, n=1):
            if math.isnan(sft_loss) or math.isinf(sft_loss) or math.isnan(distill_loss) or math.isinf(distill_loss):
                print("Skipping update due to NaN or Inf value.")
                self.count += 1
                return
            if n % 100 == 0:
                self.sft_loss = 0
                self.distill_loss = 0
                self.unmasked_rate = 0
            self.sft_loss += sft_loss
            self.distill_loss += distill_loss
            if unmasked_rate is not None:
                self.unmasked_rate += unmasked_rate
            self.count += n

        def compute(self):
            if self.count == 0:
                return 0, 0
            if model_args.enable_edit_kd:
                return self.sft_loss / self.count, self.distill_loss / self.count, self.unmasked_rate / self.count
            else:
                return self.sft_loss / self.count, self.distill_loss / self.count

    class DistillationTrainer(Trainer):
        def __init__(self, *args, teacher_models=None, tokenizers=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.student_tokenizer = tokenizers[0]
            self.teacher_tokenizer = tokenizers[1]
            self.teachers = teacher_models
            self.padding_id = -100
            self.distill_metrics = CustomMetric()
            if model_args.teacher_to_student_id_mapping is not None:
                self.tea2stu_id_mapping = json.load(open(model_args.teacher_to_student_id_mapping))
                self.stu2tea_id_mapping = torch.zeros(self.student_tokenizer.vocab_size + 256, dtype=torch.long)
                for tea_id in self.tea2stu_id_mapping:
                    if self.tea2stu_id_mapping[tea_id] != 0:
                        self.stu2tea_id_mapping[self.tea2stu_id_mapping[tea_id]] = int(tea_id)

                self.tea2stu_id_mapping = list(self.tea2stu_id_mapping.values())
                tea_vocab_size = self.teacher_tokenizer.vocab_size + len(self.teacher_tokenizer.added_tokens_decoder)
                if len(self.tea2stu_id_mapping) != tea_vocab_size:
                    self.tea2stu_id_mapping += [0] * (tea_vocab_size - len(self.tea2stu_id_mapping))
                self.tea2stu_id_mapping = torch.LongTensor(self.tea2stu_id_mapping).to(training_args.device)
                self.stu2tea_id_mapping = torch.LongTensor(self.stu2tea_id_mapping).to(training_args.device)
                self.stu2tea_id_mapping_tea = torch.LongTensor(torch.arange(self.stu2tea_id_mapping.shape[0])).to(training_args.device)
                self.stu2tea_id_mapping_stu = copy.deepcopy(self.stu2tea_id_mapping)
                self.em_tea2stu_id_mapping = copy.deepcopy(self.tea2stu_id_mapping)
                self.em_stu2tea_id_mapping = copy.deepcopy(self.stu2tea_id_mapping)

            for teacher in self.teachers:
                teacher = teacher.half().eval()
                self._move_model_to_device(teacher, training_args.device)

        def compute_loss(self, model, inputs, return_outputs=False):
            outputs_student = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels'],
                output_hidden_states=True,
                output_attentions=True
            )
            student_loss = outputs_student.loss

            with torch.no_grad():
                outputs_teacher = self.teachers[0](
                    input_ids=inputs['teacher_input_ids0'],
                    attention_mask=inputs['teacher_attention_mask0'],
                    output_hidden_states=True,
                    output_attentions=True
                )

            edit_kd_loss, unmasked_rate = self.compute_edit_distance_kd_loss(outputs_student, outputs_teacher, inputs)
            kd_loss = edit_kd_loss
            self.distill_metrics.update(
                sft_loss=student_loss.item(),
                distill_loss=kd_loss.item(),
                unmasked_rate=unmasked_rate
            )
            loss = model_args.kd_alpha * student_loss + (1.0 - model_args.kd_alpha) * kd_loss
            return (loss, outputs_student) if return_outputs else loss

        def compute_cross_entropy_loss(self, logits, target):
            loss = F.cross_entropy(logits.squeeze(0), target.squeeze(0), reduction='mean')
            nll_loss = loss
            return loss, nll_loss

        def dist_func(self, logits, teacher_logits, target=None, reduction=None):
            lprobs = torch.log_softmax(logits / model_args.kd_temperature, -1, dtype=torch.float32)
            teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
            teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32) * (model_args.kd_temperature ** 2)
            kld = (teacher_probs * (teacher_lprobs - lprobs))
            inf_mask = kld.isinf()
            kld = kld.masked_fill_(inf_mask, 0.0)
            kld = kld.sum() / torch.sum(~inf_mask).item()
            return kld

        def compute_edit_distance_kd_loss(self, outputs_student, outputs_teacher, inputs):
            target = inputs["labels"]                   # student MLM labels (-100 for unmasked tokens)
            teacher_target = inputs["teacher_labels0"]  # -100 for teacher prefix (nếu có)

            student_logits = outputs_student.logits
            stu_tokenizer = self.student_tokenizer
            tea_tokenizer = self.teacher_tokenizer

            bsz = target.shape[0]
            aligned_tea_logits = []
            aligned_stu_logits = []
            for i in range(bsz):
                # Với MLM, positions có label != -100 là các token bị mask (cần dự đoán)
                stu_content_idx = torch.nonzero(target[i].ne(self.padding_id)).view(-1)
                tea_content_idx = torch.nonzero(teacher_target[i].ne(self.padding_id)).view(-1)

                if stu_content_idx.numel() == 0 or tea_content_idx.numel() == 0:
                    # không có vị trí hợp lệ để KD
                    continue

                stu_input_ids = inputs["input_ids"][i, stu_content_idx]
                tea_input_ids = inputs["teacher_input_ids0"][i, tea_content_idx]

                try:
                    stu_per_step_logits = student_logits[i, stu_content_idx, :]
                    tea_per_step_logits = outputs_teacher.logits[i, tea_content_idx, :]
                    if stu_per_step_logits.shape[-1] == 0 or tea_per_step_logits.shape[-1] == 0:
                        continue

                    aligned_tea_content_per_step_logits, meaned_stu_content_logits, unmask_rate = self.transform_step_logits_fast(
                        stu_tokenizer,
                        tea_tokenizer,
                        stu_input_ids,
                        stu_per_step_logits,
                        tea_input_ids,
                        tea_per_step_logits,
                    )
                except Exception as e:
                    # print("align error:", e)
                    continue

                aligned_stu_logits.append(meaned_stu_content_logits)
                aligned_tea_logits.append(aligned_tea_content_per_step_logits)

            if len(aligned_stu_logits) == 0:
                return torch.tensor(0.0, device=student_logits.device), 0.0

            aligned_tea_logits = torch.stack(aligned_tea_logits, 0)
            aligned_stu_logits = torch.stack(aligned_stu_logits, 0)
            in_len = aligned_stu_logits.shape[1]
            kd_loss = self.dist_func(
                aligned_stu_logits,
                aligned_tea_logits,
                inputs["labels"][:, -in_len:],
                reduction='mean'
            )
            return kd_loss, unmask_rate

        def merge_tensor(self, values, mapping_list):
            merged_values = []
            for ids in mapping_list:
                merged_values.append(values[ids].mean(dim=0))
            merged_values = torch.stack(merged_values, dim=0)
            return merged_values

        def transform_step_logits_fast(
            self,
            base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
            blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
            base_model_input_ids: torch.LongTensor,
            base_model_per_step_logits: torch.FloatTensor,
            blending_model_input_ids: torch.LongTensor,
            blending_model_per_step_logits: torch.FloatTensor,
        ):
            """Align logits qua DTW + dynamic vocab mapping."""
            base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
            base_model_tokens = [base_model_tokenizer.convert_tokens_to_string([tok]) for tok in base_model_tokens]
            blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(blending_model_input_ids)
            blending_model_tokens = [blending_model_tokenizer.convert_tokens_to_string([tok]) for tok in blending_model_tokens]

            base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN.get(base_model_tokenizer.__class__, "Ġ")
            blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN.get(blending_model_tokenizer.__class__, "_")

            specTok_mapper = {
                '</s>': '<|im_end|>',
                '<|endoftext|>': '<|endoftext|>'
            }

            def dist_fn(a, b):
                if a in specTok_mapper and b in specTok_mapper.values():
                    return 0.0
                if b in specTok_mapper and a in specTok_mapper.values():
                    return 0.0
                aa = a.replace(blending_model_special_token, "").replace(" ", "")
                bb = b.replace(base_model_special_token, "").replace(" ", "")
                dist = editdistance.eval(aa, bb)
                if len(aa) == len(bb) == 0:
                    return 0.0
                dist = dist / (len(aa) + len(bb))
                return dist

            def cost_fn(a, b):
                if a in specTok_mapper and b in specTok_mapper.values():
                    return 0.0
                if b in specTok_mapper and a in specTok_mapper.values():
                    return 0.0
                aa = a.replace(blending_model_special_token, "").replace(" ", "")
                bb = b.replace(base_model_special_token, "").replace(" ", "")
                dist = editdistance.eval(aa, bb)
                return dist

            blending_dist_factor = calculate_weight(blending_model_per_step_logits)
            base_dist_factor = calculate_weight(base_model_per_step_logits)

            _, _, blending_to_base, base_to_blending, _ = self.dtw(
                blending_model_tokens, base_model_tokens, blending_dist_factor, base_dist_factor, norm_func=cost_fn
            )

            # Gộp logits theo mapping
            blending_model_per_step_logits = self.merge_tensor(blending_model_per_step_logits, base_to_blending)

            cnt_merge_blending_to_base = []
            for ids in blending_to_base:
                if ids not in cnt_merge_blending_to_base:
                    cnt_merge_blending_to_base.append(ids)
            blending_model_per_step_logits = self.merge_tensor(blending_model_per_step_logits, cnt_merge_blending_to_base)
            base_model_per_step_logits = self.merge_tensor(base_model_per_step_logits, cnt_merge_blending_to_base)

            topK = model_args.topk
            blending_topk_ids = torch.topk(blending_model_per_step_logits, topK).indices
            base_topk_ids = torch.topk(base_model_per_step_logits, topK).indices

            blending_topk_tokens = [[blending_model_tokenizer.decode(i) for i in ids] for ids in blending_topk_ids]
            base_topk_tokens = [[base_model_tokenizer.decode(i) for i in ids] for ids in base_topk_ids]

            tea2stu_mapper = self.tea2stu_id_mapping

            def get_dynamic_mapper(
                blending_topk_ids,
                base_topk_ids,
                blending_topk_tokens,
                base_topk_tokens,
                blending2base_mapper=tea2stu_mapper,
                em_mapper=self.em_tea2stu_id_mapping,
            ):
                dist_threashold = model_args.simi_threadshold
                em_converted_base_topk_ids = blending2base_mapper[blending_topk_ids]
                miss_hit_mask = torch.eq(em_converted_base_topk_ids, 0)

                unmapped_blending_list = []
                for pos in torch.nonzero(miss_hit_mask):
                    unmapped_blending_list.append(blending_topk_ids[pos[0]][pos[1]])

                unmapped_blending_tokens = [blending_topk_tokens[pos[0]][pos[1]] for pos in torch.nonzero(miss_hit_mask)]
                candidate_list = [base_topk_ids[pos[0]] for pos in torch.nonzero(miss_hit_mask)]
                candidate_tokens = [base_topk_tokens[pos[0]] for pos in torch.nonzero(miss_hit_mask)]

                matched_ids = torch.nonzero(torch.eq(blending2base_mapper.squeeze(0), 0)).reshape(-1).tolist()
                matched_set = set(matched_ids)
                if dist_threashold > 0.0001:
                    for id_, token, cand_ids, cand_toks in zip(unmapped_blending_list, unmapped_blending_tokens, candidate_list, candidate_tokens):
                        if em_mapper[id_] != 0:
                            continue
                        cand_ids = cand_ids.tolist()
                        cand_mapper = {tid: tok for tok, tid in zip(cand_toks, cand_ids)}
                        cand_ids = list(set(cand_ids).difference(matched_set))
                        if len(cand_ids) == 0:
                            continue
                        min_dist = 1e9
                        simi_id = 0
                        for cand_id in cand_ids:
                            cand_tok = cand_mapper[cand_id]
                            tok_dist = dist_fn(token, cand_tok)
                            if tok_dist < dist_threashold and tok_dist < min_dist:
                                simi_id = cand_id
                                min_dist = tok_dist
                        if simi_id != 0:
                            blending2base_mapper[id_] = simi_id

                converted_base_topk_ids = blending2base_mapper[blending_topk_ids].to(blending_model_per_step_logits.device)
                unmatch_mask = torch.eq(converted_base_topk_ids, 0)
                masked_blending_topk_ids = blending_topk_ids.masked_fill_(unmatch_mask, 0)
                return converted_base_topk_ids, masked_blending_topk_ids

            base_logits = []
            blending_logits = []

            # Teacher->Student
            stu_converted_topk_ids, tea_converted_topk_ids = get_dynamic_mapper(
                blending_topk_ids, base_topk_ids, blending_topk_tokens, base_topk_tokens,
                blending2base_mapper=copy.deepcopy(self.tea2stu_id_mapping),
                em_mapper=self.em_tea2stu_id_mapping,
            )
            stu_model_per_step_logits = base_model_per_step_logits.gather(-1, stu_converted_topk_ids)
            tea_model_per_step_logits = blending_model_per_step_logits.gather(-1, tea_converted_topk_ids)

            stu_logit_mask = stu_converted_topk_ids.eq(0)
            tea_logit_mask = tea_converted_topk_ids.eq(0)
            stu_model_per_step_logits.masked_fill_(stu_logit_mask, -10000.0)
            tea_model_per_step_logits.masked_fill_(tea_logit_mask, -10000.0)
            mask_rate = stu_logit_mask.sum().item() / (stu_logit_mask.size(0) * stu_logit_mask.size(1))

            base_logits.append(tea_model_per_step_logits)
            blending_logits.append(stu_model_per_step_logits)

            # Student->Teacher
            tea_converted_topk_ids, stu_converted_topk_ids = get_dynamic_mapper(
                base_topk_ids, blending_topk_ids, base_topk_tokens, blending_topk_tokens,
                blending2base_mapper=copy.deepcopy(self.stu2tea_id_mapping),
                em_mapper=self.em_stu2tea_id_mapping,
            )
            stu_model_per_step_logits_2 = base_model_per_step_logits.gather(-1, stu_converted_topk_ids)
            tea_model_per_step_logits_2 = blending_model_per_step_logits.gather(-1, tea_converted_topk_ids)

            stu_logit_mask2 = stu_converted_topk_ids.eq(0)
            tea_logit_mask2 = tea_converted_topk_ids.eq(0)
            stu_model_per_step_logits_2.masked_fill_(stu_logit_mask2, -10000.0)
            tea_model_per_step_logits_2.masked_fill_(tea_logit_mask2, -10000.0)
            mask_rate += stu_logit_mask2.sum().item() / (stu_logit_mask2.size(0) * stu_logit_mask2.size(1))
            mask_rate = mask_rate / 2

            base_logits.append(stu_model_per_step_logits_2)
            blending_logits.append(tea_model_per_step_logits_2)

            return torch.cat(base_logits, dim=-1), torch.cat(blending_logits, dim=-1), 1 - mask_rate

        # @timer
        def dtw(self, series_1, series_2, series1_factor, series2_factor, norm_func=np.linalg.norm):
            matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
            matrix[0, :] = np.inf
            matrix[:, 0] = np.inf
            matrix[0, 0] = 0
            for i, (vec1, fc1) in enumerate(zip(series_1, series1_factor)):
                for j, (vec2, fc2) in enumerate(zip(series_2, series2_factor)):
                    cost = norm_func(vec1, vec2) * fc1 * fc2
                    matrix[i + 1, j + 1] = cost + min(matrix[i, j + 1], matrix[i + 1, j], matrix[i, j])
            matrix = matrix[1:, 1:]
            i = matrix.shape[0] - 1
            j = matrix.shape[1] - 1
            matches = []
            mappings_series_1 = [list() for _ in range(matrix.shape[0])]
            mappings_series_2 = [list() for _ in range(matrix.shape[1])]
            while i > 0 or j > 0:
                matches.append((i, j))
                mappings_series_1[i].append(j)
                mappings_series_2[j].append(i)
                option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
                option_up = matrix[i - 1, j] if i > 0 else np.inf
                option_left = matrix[i, j - 1] if j > 0 else np.inf
                move = np.argmin([option_diag, option_up, option_left])
                if move == 0:
                    i -= 1
                    j -= 1
                elif move == 1:
                    i -= 1
                else:
                    j -= 1
            matches.append((0, 0))
            mappings_series_1[0].append(0)
            mappings_series_2[0].append(0)
            matches.reverse()
            for mp in mappings_series_1:
                mp.reverse()
            for mp in mappings_series_2:
                mp.reverse()
            return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix

        def log(self, logs):
            if not model_args.enable_edit_kd:
                sft_loss, distill_loss = self.distill_metrics.compute()
                logs.update({'sft_loss': sft_loss, 'distill_loss': distill_loss})
            else:
                sft_loss, distill_loss, unmasked_rate = self.distill_metrics.compute()
                logs.update({'sft_loss': sft_loss, 'distill_loss': distill_loss, 'unmasked_rate': unmasked_rate})
            super().log(logs)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # Load dataset
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            if len(data_args.train_file.split(',')) > 0:
                data_files["train"] = data_args.train_file.split(',')
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=None,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_verifications=False,
            **dataset_args,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # Config & tokenizer
    config_kwargs = {"revision": model_args.model_revision, "use_auth_token": True if model_args.use_auth_token else None}
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, trust_remote_code=True, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. Not supported by this script."
        )

    # Load models
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        if model_args.student_arch == "masked_lm":
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                config=config if config is not None else None,
                trust_remote_code=True,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                trust_remote_code=True,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        if training_args.gradient_checkpointing:
            print("Setting enable_input_require_grads")
            model.enable_input_require_grads()

        teachers = []
        teacher_tokenizers = []
        for teacher_path in model_args.teacher_path.split(','):
            teacher = AutoModelForCausalLM.from_pretrained(
                teacher_path,
                trust_remote_code=True,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_path,
                trust_remote_code=True
            )
            if teacher_tokenizer.pad_token is None and teacher_tokenizer.eos_token is not None:
                teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
            teachers.append(teacher)
            teacher_tokenizers.append(teacher_tokenizer)

        print("freeze_layers:", model_args.freeze_layers)
        print('freeze_emb', model_args.freeze_emb)
        if model_args.freeze_layers is not None or model_args.freeze_emb is not None:
            for name, param in model.named_parameters():
                param.requires_grad = True
                if name == "model.embed_tokens.weight" and model_args.freeze_emb:
                    param.requires_grad = False
                if model_args.freeze_layers is not None:
                    for layer in model_args.freeze_layers.split(','):
                        if f"model.layers.{layer}." in name:
                            param.requires_grad = False
                            print("freezing param:", name)
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # Padding token & side
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    tokenizer.padding_side = "right" if model_args.student_arch == "masked_lm" else "left"
    print("debug: tokenizer.padding_side =", tokenizer.padding_side)

    # Resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Columns
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Block size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "Tokenizer model_max_length > 1024; default block_size=1024. Override with --block_size if needed."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"block_size ({data_args.block_size}) > tokenizer.model_max_length ({tokenizer.model_max_length}). "
                f"Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
        print("debug: the actual block_size is ", block_size, "   tokenizer.model_max_length=", tokenizer.model_max_length)

    # Preprocess
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    is_mlm_student = (model_args.student_arch == "masked_lm")

    def preprocess_function(examples):
        with CaptureLogger(tok_logger) as cl:
            padding = False
            text = examples[text_column_name]

            if "prefix" in column_names:
                prefix = examples["prefix"]
                stu_text = [s + t + ("" if is_mlm_student else (tokenizer.eos_token or "")) for s, t in zip(prefix, text)]

                prefix_tok = tokenizer(prefix, truncation=True, max_length=block_size, padding=False)
                stu_tok   = tokenizer(stu_text, truncation=True, max_length=block_size, padding=padding)

                teacher_tokenizer = teacher_tokenizers[0]
                if teacher_tokenizer.pad_token is None and teacher_tokenizer.eos_token is not None:
                    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
                tea_text = [s + t + (teacher_tokenizer.eos_token or "") for s, t in zip(prefix, text)]
                tea_prefix_tok = teacher_tokenizer(prefix, truncation=True, max_length=block_size, padding=False)
                tea_tok        = teacher_tokenizer(tea_text, truncation=True, max_length=block_size, padding=padding)

                if not is_mlm_student:
                    labels = copy.deepcopy(stu_tok["input_ids"])
                    prefix_lens = [len(p) for p in prefix_tok["input_ids"]]
                    for lab, p_len in zip(labels, prefix_lens):
                        lab[:p_len] = [IGNORE_INDEX] * p_len
                    stu_tok["labels"] = labels

                tea_labels = copy.deepcopy(tea_tok["input_ids"])
                tea_prefix_lens = [len(p) for p in tea_prefix_tok["input_ids"]]
                for lab, p_len in zip(tea_labels, tea_prefix_lens):
                    lab[:p_len] = [IGNORE_INDEX] * p_len

                stu_tok["teacher_labels0"] = tea_labels
                stu_tok["teacher_input_ids0"] = tea_tok["input_ids"]
                stu_tok["teacher_attention_mask0"] = tea_tok["attention_mask"]
                text_tokenized = stu_tok

            elif "input" in column_names:
                prefix = [ins + '\n### Input:\n' + inp for ins, inp in zip(examples['instruction'], examples['input'])]
                out_text = [p + "\n### Output:\n" + o for p, o in zip(prefix, examples['output'])]
                stu_text = [t + ("" if is_mlm_student else (tokenizer.eos_token or "")) for t in out_text]

                prefix_tok = tokenizer(prefix, truncation=True, max_length=block_size, padding=False)
                stu_tok    = tokenizer(stu_text, truncation=True, max_length=block_size, padding=padding)

                if not is_mlm_student:
                    labels = copy.deepcopy(stu_tok["input_ids"])
                    prefix_lens = [len(p) for p in prefix_tok["input_ids"]]
                    for lab, p_len in zip(labels, prefix_lens):
                        lab[:p_len] = [IGNORE_INDEX] * p_len
                    stu_tok["labels"] = labels

                teacher_tokenizer = teacher_tokenizers[0]
                if teacher_tokenizer.pad_token is None and teacher_tokenizer.eos_token is not None:
                    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
                tea_text = [t + (teacher_tokenizer.eos_token or "") for t in out_text]
                tea_tok  = teacher_tokenizer(tea_text, truncation=True, max_length=block_size, padding=padding)

                # Ở nhánh này không có prefix riêng cho teacher; dùng full labels (hoặc sau collator pad -100)
                stu_tok["teacher_labels0"] = copy.deepcopy(tea_tok["input_ids"])
                stu_tok["teacher_input_ids0"] = tea_tok["input_ids"]
                stu_tok["teacher_attention_mask0"] = tea_tok["attention_mask"]
                text_tokenized = stu_tok

            else:
                # plain text
                stu_text = [t + ("" if is_mlm_student else (tokenizer.eos_token or "")) for t in text]
                stu_tok  = tokenizer(stu_text, truncation=True, max_length=block_size, padding=padding)

                if not is_mlm_student:
                    labels = copy.deepcopy(stu_tok["input_ids"])
                    stu_tok["labels"] = labels

                teacher_tokenizer = teacher_tokenizers[0]
                if teacher_tokenizer.pad_token is None and teacher_tokenizer.eos_token is not None:
                    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
                tea_text = [t + (teacher_tokenizer.eos_token or "") for t in text]
                tea_tok  = teacher_tokenizer(tea_text, truncation=True, max_length=block_size, padding=padding)

                stu_tok["teacher_labels0"] = copy.deepcopy(tea_tok["input_ids"])
                stu_tok["teacher_input_ids0"] = tea_tok["input_ids"]
                stu_tok["teacher_attention_mask0"] = tea_tok["attention_mask"]
                text_tokenized = stu_tok

        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return text_tokenized

    with training_args.main_process_first(desc="example per line with padding"):
        lm_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc=f"Tokenize with padding",
        )

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        print(train_dataset)
        if not data_args.streaming:
            train_dataset = train_dataset.shuffle(seed=training_args.seed)
        else:
            train_dataset = train_dataset.shuffle(seed=training_args.seed)

        logger.info("xxx: Showcase the tokenized training samples.")
        tmp_idx = 0
        for tmp_example in train_dataset:
            if tmp_idx > 3:
                break
            tmp_idx += 1
            print(tmp_example)

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            try:
                return {"accuracy": torch.sum(preds == labels) / len(labels)}
            except:
                return {"accuracy": (torch.sum(torch.tensor(preds) == torch.tensor(labels)).to(device=training_args.device) / len(labels))}
    else:
        preprocess_logits_for_metrics = None
        compute_metrics = None

    training_args.remove_unused_columns = False

    # Chọn collator & metric theo student_arch
    if model_args.student_arch == "masked_lm":
        data_collator = CDMMLMCollator(
            student_tokenizer=tokenizer,
            teacher_tokenizer=teacher_tokenizers[0],
            mlm_probability=model_args.mlm_probability,
        )
        cm = None
        plfm = None
    else:
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True, label_pad_token_id=IGNORE_INDEX
        )
        cm = compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None
        plfm = preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None

    trainer = DistillationTrainer(
        model=model,
        teacher_models=teachers,
        tokenizers=(tokenizer, teacher_tokenizers[0]),
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=cm,
        preprocess_logits_for_metrics=plfm,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            if training_args.resume_from_checkpoint == "True":
                checkpoint = True
            else:
                checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
