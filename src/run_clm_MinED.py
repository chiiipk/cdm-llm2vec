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
import torch.nn.functional as F
import torch
from datasets import load_dataset
import transformers
print(transformers.__version__)
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    # Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from my_trainer import Trainer
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version
# xxx: 2023-03-21
import copy
import torch.nn as nn
import torch.nn.functional as F
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result  # Return the result of the original function
    return wrapper

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

import editdistance
TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.LlamaTokenizerFast: "▁",
    transformers.GPTNeoXTokenizerFast: "Ġ",
    transformers.GPT2Tokenizer: "Ġ",
    transformers.GPT2TokenizerFast: "Ġ",
    transformers.Qwen2Tokenizer: "Ġ",
    transformers.Qwen2TokenizerFast: "Ġ",
}

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# xxx: 2023-03-21
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
    factor = torch.sigmoid((entropy - entropy_min)/(entropy_max-entropy_min) * 4 - 2) # [0,1]\
    factor = torch.tensor((factor * 3 + 3), dtype=torch.int32).detach().cpu().tolist()

    return factor


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    teacher_path: Optional[str] = field(
        default=None,
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    freeze_emb: bool = field(
        default=False,
        metadata={"help": "whether freeze the weights of emb"},
    )

    freeze_layers: Optional[str] = field(
        default=None,
        metadata={"help": "whether freeze the weights of some layers"},
    )

    kd_alpha: Optional[float] = field(
        default=0.9
    )
    kd_temperature: Optional[float] = field(
        default=0.8
    )
    enable_edit_kd: Optional[bool] = field(
        default=False,
    )
    enable_topk: Optional[bool] = field(
        default=False,
    )
    topk: Optional[int] = field(
        default=100,
    )
    simi_threadshold: Optional[float] = field(
        default=0.3,
    )
    teacher_to_student_id_mapping: Optional[str] = field(
        default=None
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    shuffle_buffer_size: int = field(default=10000, metadata={"help": "Enable streaming mode"})

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    padding_side: str = field(
        default=None, metadata={"help": "padding_side"}
    )

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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    training_args.disable_tqdm = True


    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
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
            if n%100==0:
                self.sft_loss = 0
                self.distill_loss = 0
                self.unmasked_rate = 0
            self.sft_loss += sft_loss
            self.distill_loss += distill_loss
            if unmasked_rate is not None:
                self.unmasked_rate += unmasked_rate
            self.count += n

        def compute(self):
            if self.count==0:
                return 0, 0
            if model_args.enable_edit_kd:
                return self.sft_loss/self.count, self.distill_loss/self.count, self.unmasked_rate/self.count
            else:
                return self.sft_loss/self.count, self.distill_loss/self.count
        

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
                
                self.stu2tea_id_mapping = torch.zeros(self.student_tokenizer.vocab_size+256, dtype=torch.long)
                for tea_id in self.tea2stu_id_mapping:

                    if self.tea2stu_id_mapping[tea_id]!=0:
                        self.stu2tea_id_mapping[self.tea2stu_id_mapping[tea_id]] = int(tea_id)

                self.tea2stu_id_mapping = list(self.tea2stu_id_mapping.values())
                tea_vocab_size = self.teacher_tokenizer.vocab_size + len(self.teacher_tokenizer.added_tokens_decoder)
                if len(self.tea2stu_id_mapping) != tea_vocab_size:
                    self.tea2stu_id_mapping += [0] * (tea_vocab_size - len(self.tea2stu_id_mapping))
                self.tea2stu_id_mapping = torch.LongTensor(self.tea2stu_id_mapping).to(training_args.device)
                self.stu2tea_id_mapping = torch.LongTensor(self.stu2tea_id_mapping).to(training_args.device)
                self.stu2tea_id_mapping_tea = torch.LongTensor(torch.arange(self.stu2tea_id_mapping.shape[0])).to(training_args.device)
                self.stu2tea_id_mapping_stu = copy.deepcopy(self.stu2tea_id_mapping)

            for teacher in self.teachers:
                # place each teacher on same device as student
                teacher.half().eval()
                self._move_model_to_device(teacher, training_args.device)
                
    
            self.model.enable_input_require_grads()
           

        
        


        def compute_loss(self, model, inputs, return_outputs=False):
            outputs_student = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'], output_hidden_states=True, output_attentions=True)
            student_loss = outputs_student.loss
            
            
            # compute teacher output
            with torch.no_grad():
                outputs_teacher = self.teachers[0](input_ids=inputs['teacher_input_ids0'], attention_mask=inputs['teacher_attention_mask0'], output_hidden_states=True, output_attentions=True)
            
            kd_loss = 0
            
            
            edit_kd_loss, unmasked_rate = self.compute_edit_distance_kd_loss(outputs_student, outputs_teacher, inputs)
            kd_loss = edit_kd_loss
            self.distill_metrics.update(sft_loss=student_loss.item(), distill_loss=(kd_loss).item(), unmasked_rate=unmasked_rate)

                
            # key

            loss = model_args.kd_alpha * student_loss + (1.0 - model_args.kd_alpha) * (kd_loss)
         
            return (loss, outputs_student) if return_outputs else loss
        

        def compute_cross_entropy_loss(self, logits, target):
            loss = F.cross_entropy(logits.squeeze(0), target.squeeze(0) ,reduction='mean')
            nll_loss = loss
            return loss, nll_loss
        

        def dist_func(
            self, 
            logits, 
            teacher_logits, 
            target=None,
            reduction=None
        ):
            # self.loss_func = torch.nn.KLDivLoss(reduction='none')
            lprobs = torch.log_softmax(logits/model_args.kd_temperature, -1, dtype=torch.float32)
            teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
            teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32) * (model_args.kd_temperature ** 2)
            kld = (teacher_probs * (teacher_lprobs - lprobs))
            inf_mask = kld.isinf()
            kld = kld.masked_fill_(inf_mask, 0.0)
            kld = kld.sum()/ torch.sum(~inf_mask).item()


            return kld
        

        def compute_edit_distance_kd_loss(
            self, outputs_student, outputs_teacher, inputs
        ):
            target = inputs["labels"]
            teacher_target = inputs[f"teacher_labels0"]
            student_logits = outputs_student.logits
            stu_tokenizer = self.student_tokenizer
            tea_tokenizer = self.teacher_tokenizer

            bsz = target.shape[0]
            aligned_tea_logits = []
            aligned_stu_logits = []
            for i in range(bsz):
                assert self.padding_id in target[i]
                stu_content_idx = torch.nonzero(target[i].ne(self.padding_id)).view(-1)
                stu_input_ids = inputs["input_ids"][i, stu_content_idx]

                tea_content_idx = torch.nonzero(teacher_target[i].ne(self.padding_id)).view(-1)
                tea_input_ids = inputs[f"teacher_input_ids0"][i, tea_content_idx]

                stu_per_step_logits = student_logits[i, stu_content_idx, :].float()
                tea_per_step_logits = outputs_teacher.logits[i, tea_content_idx, :].float()   
                if stu_per_step_logits.shape[-1]==0 or tea_per_step_logits.shape[-1]==0:
                    return torch.Tensor([0.0]).to(student_logits.device), 0.0
                aligned_tea_content_per_step_logits, meaned_stu_content_logits, unmask_rate = self.transform_step_logits_fast(
                    stu_tokenizer,
                    tea_tokenizer,
                    stu_input_ids,
                    stu_per_step_logits,
                    tea_input_ids,
                    tea_per_step_logits,
                )
                aligned_stu_logits.append(meaned_stu_content_logits)
                aligned_tea_logits.append(aligned_tea_content_per_step_logits)
                
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
                # merged_values.append(values[ids[0]])
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
            """faster implementation to align logits"""
            base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
            base_model_tokens = [base_model_tokenizer.convert_tokens_to_string([tok]) for tok in base_model_tokens]
            blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
                blending_model_input_ids
            )
            blending_model_tokens = [blending_model_tokenizer.convert_tokens_to_string([tok]) for tok in blending_model_tokens]
            if base_model_tokenizer.__class__ not in TOKENIZER_TO_SPECIAL_TOKEN:
                # print("Warning: not implemented for base tokenizer!", base_model_tokenizer)
                base_model_special_token = "Ġ"
            else:
                base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
                    base_model_tokenizer.__class__
                ]
            if blending_model_tokenizer.__class__ not in TOKENIZER_TO_SPECIAL_TOKEN:
                # print("Warning: not implemented for blending tokenizer!", blending_model_tokenizer)
                blending_model_special_token = '_'
            else:
                blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
                    blending_model_tokenizer.__class__
                ]
            # revise @xxx: add special tokens to the special tokens map
            specTok_mapper = {
                '</s>': '<|im_end|>',
                '<|endoftext|>':'<|endoftext|>'
            }

            def dist_fn(a, b):
                """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
                if a in specTok_mapper and b in specTok_mapper.values():
                    return 0.0
                if b in specTok_mapper and a in specTok_mapper.values():
                    return 0.0
                aa = a.replace(blending_model_special_token, "")
                bb = b.replace(base_model_special_token, "")
                aa = a.replace(" ", "")
                bb = b.replace(" ", "")
                dist = editdistance.eval(aa, bb) 
                if len(aa)==len(bb)==0:
                    return 0.0
                dist = dist / (len(aa)+len(bb))
                return dist
           
            def cost_fn(a, b):
                """cost function for sequence alignment"""
                if a in specTok_mapper and b in specTok_mapper.values():
                    return 0.0
                if b in specTok_mapper and a in specTok_mapper.values():
                    return 0.0
                aa = a.replace(blending_model_special_token, "")
                bb = b.replace(base_model_special_token, "")
                aa = a.replace(" ", "")
                bb = b.replace(" ", "")
                dist = editdistance.eval(aa, bb)
                return dist
            
            blending_dist_factor = calculate_weight(blending_model_per_step_logits) 
            base_dist_factor = calculate_weight(base_model_per_step_logits)
            # obtain sequence token alignment (each stu token to which tea token)
            _, _, blending_to_base, base_to_blending, _ = self.dtw(
                blending_model_tokens, base_model_tokens, blending_dist_factor, base_dist_factor, norm_func=cost_fn
            ) 

            merged_blending_tokens = []
            for ids in base_to_blending:
                merged_token = ''
                for id in ids:
                    merged_token += blending_model_tokens[id]
                merged_blending_tokens.append(merged_token)
            
            
            
            blending_model_per_step_logits = self.merge_tensor(
                blending_model_per_step_logits,
                base_to_blending
            )
            cnt_merge_blending_to_base = []
            for ids in blending_to_base:
                if ids not in cnt_merge_blending_to_base:
                    cnt_merge_blending_to_base.append(ids)
            blending_model_per_step_logits = self.merge_tensor(
                blending_model_per_step_logits,
                cnt_merge_blending_to_base
            )
            base_model_per_step_logits = self.merge_tensor(
                base_model_per_step_logits,
                cnt_merge_blending_to_base
            )
            
            topK = model_args.topk
            blending_topk_ids = torch.topk(blending_model_per_step_logits, topK).indices
            base_topk_ids = torch.topk(base_model_per_step_logits, topK).indices
            
            blending_topk_tokens = []
            for ids in blending_topk_ids:
                blending_topk_tokens.append([blending_model_tokenizer.decode(id) for id in ids])
            
            base_topk_tokens = []
            for ids in base_topk_ids:
                base_topk_tokens.append([base_model_tokenizer.decode(id) for id in ids])

            tea2stu_mapper = self.tea2stu_id_mapping
            def get_dymaic_mapper(
                    blending_topk_ids, 
                    base_topk_ids, 
                    blending_topk_tokens, 
                    base_topk_tokens, 
                    blending2base_mapper=tea2stu_mapper,
                ):
                dist_threashold = model_args.simi_threadshold
                # get the exact matching result
                em_converted_base_topk_ids = blending2base_mapper[blending_topk_ids]
                # get the elements that are not exact match use a mask with 0 judgement
                miss_hit_mask = torch.eq(em_converted_base_topk_ids, 0)
                # get the unmapped base tokens, and the correspondent candidate tokens in teacher
                unmapped_blending_list = []
                # [base_topk_ids[pos] for pos in torch.nonzero(miss_hit_mask)]
                for pos in torch.nonzero(miss_hit_mask): unmapped_blending_list.append(blending_topk_ids[pos[0]][pos[1]])

                unmapped_blending_tokens = [blending_topk_tokens[pos[0]][pos[1]] for pos in torch.nonzero(miss_hit_mask)]
                candidate_list = [base_topk_ids[pos[0]] for pos in torch.nonzero(miss_hit_mask)]
                candidate_tokens = [base_topk_tokens[pos[0]] for pos in torch.nonzero(miss_hit_mask)]
                # traversal to get the supplemental mapping pairs.
                matched_ids = torch.nonzero(torch.eq(blending2base_mapper.squeeze(0), 0)).reshape(-1).tolist()
                matched_set = set(matched_ids)
                if dist_threashold >0.0001:
                    for id, token, cand_ids, cand_toks in zip(unmapped_blending_list, unmapped_blending_tokens, candidate_list, candidate_tokens):
                        if blending2base_mapper[id]!=0:
                            continue
                        cand_ids = cand_ids.tolist()
                        cand_mapper = {tid:tok for tok, tid in zip(cand_toks, cand_ids)}
                        cand_ids = list(set(cand_ids).difference(matched_set))
                        if len(cand_ids)==0:
                            continue
                        min_dist = 1000
                        simi_id = 0
                        for cand_id in cand_ids:
                            cand_tok = cand_mapper[cand_id]
                            tok_dist = dist_fn(token, cand_tok)
                            if tok_dist<dist_threashold and tok_dist<min_dist:
                                simi_id = cand_id
                                min_dist = tok_dist
                        if simi_id!=0:
                            # update the mapper, keep the life cycle in the whole training step
                            blending2base_mapper[id] = simi_id
                        
                converted_base_topk_ids = blending2base_mapper[blending_topk_ids].to(blending_model_per_step_logits.device)
                unmatch_mask = torch.eq(converted_base_topk_ids, 0)
                masked_blending_topk_ids = blending_topk_ids.masked_fill_(unmatch_mask, 0)
                return blending2base_mapper

            # this block, convert the student token id to map the teacher top 100
            base_logits = []
            blending_logits = []
            tmp_mapper = get_dymaic_mapper(blending_topk_ids, base_topk_ids, blending_topk_tokens, base_topk_tokens, blending2base_mapper=self.tea2stu_id_mapping)
            stu_model_per_step_logits = blending_model_per_step_logits[:, tmp_mapper]
            tmp_logit_mask = tmp_mapper.eq(0).repeat(stu_model_per_step_logits.shape[0], 1)  # repeat to logits shape          
            stu_model_per_step_logits.masked_fill_(tmp_logit_mask, -10000.0)         
    
            mask_rate = torch.ne(tmp_mapper, 0).sum().item() / tmp_mapper.size(0)

            base_logits.append(blending_model_per_step_logits[:, :stu_model_per_step_logits.shape[-1]])
            blending_logits.append(stu_model_per_step_logits[:, :blending_model_per_step_logits.shape[-1]])



            return torch.cat(base_logits, dim=-1), torch.cat(blending_logits, dim=-1), 1-mask_rate


        # @timer
        def dtw(self, series_1, series_2, series1_factor, series2_factor, norm_func=np.linalg.norm):
            matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
            matrix[0, :] = np.inf
            matrix[:, 0] = np.inf
            matrix[0, 0] = 0
            for i, (vec1, fc1) in enumerate(zip(series_1, series1_factor)):
                for j, (vec2, fc2) in enumerate(zip(series_2, series2_factor)):
                    cost = norm_func(vec1, vec2) * fc1 * fc2
                    
                    # cost = norm_func(vec1, vec2)
                    matrix[i + 1, j + 1] = cost + min(
                        matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
                    )
            matrix = matrix[1:, 1:]
            i = matrix.shape[0] - 1
            j = matrix.shape[1] - 1
            matches = []
            mappings_series_1 = [list() for v in range(matrix.shape[0])]
            mappings_series_2 = [list() for v in range(matrix.shape[1])]
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
                logs.update({
                    'sft_loss': sft_loss,
                    'distill_loss': distill_loss,
                })
            else:
                sft_loss, distill_loss, unmasked_rate = self.distill_metrics.compute()
                logs.update({
                    'sft_loss': sft_loss,
                    'distill_loss': distill_loss,
                    'unmasked_rate': unmasked_rate
                })
            
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
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
        print(data_files)
        print(dataset_args)
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=None,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_verifications=False,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
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



    config_kwargs = {
        # "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
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
        # "cache_dir": model_args.cache_dir,
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
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            trust_remote_code=True,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # model.config.use_cache = False
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
            teachers.append(teacher)
            teacher_tokenizers.append(teacher_tokenizer)
   






    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    # model = torch.compile(model)
    # xxx: 2023-03-21, add padding
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    # if data_args.padding_side is not None:
    tokenizer.padding_side = "left"
        # tokenizer.padding_side = data_args.padding_side
    print("debug: tokenizer.padding_side =", tokenizer.padding_side )    

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]



    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
        print("debug: the actual block_size is ", block_size, "   tokenizer.model_max_length=", tokenizer.model_max_length)


    # xxx: 2023-03-14
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    def preprocess_function(examples):
        with CaptureLogger(tok_logger) as cl:
            # padding = "max_length"  # or False
            padding = False
            text = examples[text_column_name]  # may have multiple strings
            if "prefix" in column_names:
                prefix = examples["prefix"] 
                text = [s + t+tokenizer.eos_token for s, t in zip(prefix, text)]
                prefix_tokenized = tokenizer(prefix, truncation=True, max_length=block_size, padding=False)
                text_tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=padding)
                labels = copy.deepcopy(text_tokenized["input_ids"])

                prefix_lengths = [len(p) for p in prefix_tokenized["input_ids"]]
                for label, prefix_len in zip(labels, prefix_lengths):  # Do not compute loss for prompt inputs
                    label[:prefix_len] = [IGNORE_INDEX] * prefix_len  # [IGNORE_INDEX for i in range(prefix_len)]
                for i, teacher_tokenizer in enumerate(teacher_tokenizers):
                    text = [t.replace(tokenizer.eos_token, teacher_tokenizer.eos_token) for t in text]
                    teacher_prefix_tokenized = teacher_tokenizer(prefix, truncation=True, max_length=block_size, padding=False)
                    teacher_text_tokenized = teacher_tokenizer(text, truncation=True, max_length=block_size, padding=padding)
                    teacher_labels = copy.deepcopy(teacher_text_tokenized["input_ids"])
                   
                    teacher_prefix_lengths = [len(p) for p in teacher_prefix_tokenized["input_ids"]]
                    for label, prefix_len in zip(teacher_labels, teacher_prefix_lengths):  
                        label[:prefix_len] = [IGNORE_INDEX] * prefix_len 
                    
                    text_tokenized[f'teacher_labels{i}'] = teacher_labels
                    text_tokenized[f'teacher_input_ids{i}'] = teacher_text_tokenized['input_ids']
                    text_tokenized[f'teacher_attention_mask{i}'] = teacher_text_tokenized['attention_mask']
            elif 'input' in column_names:
                prefix = [ins +'\n### Input:\n'+ inp for ins, inp in zip(examples['instruction'], examples['input'])]
                text = [p + "\n### Output:\n"+ o + tokenizer.eos_token for p, o in zip(prefix, examples['output'])]
                prefix_tokenized = tokenizer(prefix, truncation=True, max_length=block_size, padding=False)
                text_tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=padding)
                labels = copy.deepcopy(text_tokenized["input_ids"])
                prefix_lengths = [len(p) for p in prefix_tokenized["input_ids"]]
                for label, prefix_len in zip(labels, prefix_lengths):  # Do not compute loss for prompt inputs
                    label[:prefix_len] = [IGNORE_INDEX] * prefix_len  # [IGNORE_INDEX for i in range(prefix_len)]

            else:
                text = [t+tokenizer.eos_token for t in text]
                text_tokenized = tokenizer(text, truncation=True, max_length=block_size, padding=False)
                labels = copy.deepcopy(text_tokenized["input_ids"])
            text_tokenized["labels"] = labels
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return text_tokenized


    # xxx: 2023-03-17
    with training_args.main_process_first(desc="example per line with padding"):
        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                desc=f"Tokenize with padding",
            )
        else:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                desc=f"Tokenize with padding",
            )


    if training_args.do_train:
        #if "train" not in tokenized_datasets:
        # xxx: 2023-03-14
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
            #train_dataset = train_dataset.shuffle(seed=training_args.seed, buffer_size=data_args.shuffle_buffer_size)        
            train_dataset = train_dataset.shuffle(seed=training_args.seed)

        # xxx: print samples
        logger.info("xxx: Showcase the tokenized training samples.")
        tmp_idx = 0
        for tmp_example in train_dataset:
            if tmp_idx > 3:
                break
            if len(tmp_example["input_ids"]) > 3000 or True:
                tmp_idx += 1
                print(tmp_example)
        #for i in range(99999):
        #    print(next(iter(train_dataset)))

    if training_args.do_eval:
        #if "validation" not in tokenized_datasets:
        # xxx: 2023-03-14
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        #metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            try:
                return {"accuracy":torch.sum(preds == labels)/len(labels)}
            except:
                return {"accuracy":(torch.sum(torch.tensor(preds) == torch.tensor(labels)).to(device=training_args.device)/len(labels))}
            #return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    training_args.remove_unused_columns = False



    trainer = DistillationTrainer(
        model=model,
        teacher_models=teachers,
        tokenizers=(tokenizer, teacher_tokenizers[0]),
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                          padding=True, label_pad_token_id=IGNORE_INDEX),
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
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
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
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
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
