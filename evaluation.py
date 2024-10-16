from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding, AdamW, get_scheduler
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
import torch
import random
import numpy as np
from utils import derive_num_from_answer, derive_num_from_output, derive_choice_from_output, get_extractors, derive_number_opencompass
from utils import COT_EXAMPLES, gsm8k_opencompass_chat, gsm8k_opencompass_prompt
from utils import SupervisedDataset
from datasets import load_dataset
import jsonlines
from tqdm import tqdm
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, gather_object
import logging
import tqdm
from peft import PeftModel

'''
This script is to evaluate the LLM's performance on test dataset
'''

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
device = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/incoming/LLM/llama2/llama2-7b")
    filter_base_model_path: str = field(default="")
    vocab_size: int = field(default=0)
    peft_model_path: str = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(
        default="/home/LAB/jiangcy/AdaDF/samples/gsm8k_test.jsonl", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    valid_data_path: str = field(
        default=None, metadata={"help": "valid data path, name:split"}
    )
    dataset_name: str = field(
        default=None
    )
    data_filter_mode: str = field(
        default="Consistency", metadata={"help": "Consistency, Groundtruth, Entropy, Weighted"}
    )
    lazy_preprocess: bool = False
    uncertainty_th: float = field(
        default=1.0
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=800,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    filter_training_batch_size: int = field(default=8)
    valid_batch_size: int = field(default=16)
    filter_training_epochs: int = field(default=10)
    filter_model_lr: float = field(
        default=1e-3
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def load_valid_data(valid_data_path):
    assert ":" in valid_data_path
    name, split = valid_data_path.split(":")
    if split.isdigit():
        dataset = load_dataset(name, data_dir="main", split="train[{}:]".format(split)) if name in [
            "gsm8k"] else load_dataset(name, split="train[{}:]".format(split))
    else:
        dataset = load_dataset(name, split=split)
    print("Load valid dataset, total length is {}".format(len(dataset)))
    return dataset


def load_test_data(tokenizer, data_args, training_args):
    dataset_path, split = data_args.eval_data_path.split(":")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if data_args.dataset_name == "facebook/anli":
        dataset = load_dataset(data_args.dataset_name, split="test_r1[:500]")
    elif data_args.dataset_name == "facebook/anli2":
        dataset = load_dataset("facebook/anli", split="test_r2[:500]")
    else:
        dataset = load_dataset(dataset_path, data_dir="main", split=split) if dataset_path in [
            "gsm8k"] else load_dataset(dataset_path, split=split)
    dp, ge, qe = get_extractors(data_args.dataset_name)

    test_dataset = SupervisedDataset(
        dataset,
        tokenizer=tokenizer,
        max_len=training_args.model_max_length,
        data_processor=dp,
        groundtruth_extractor=ge,
        question_extractor=qe
    )
    return test_dataset


def load_train_data(tokenizer: transformers.PreTrainedTokenizer, data_args, max_len, weights):
    train_data = []
    with jsonlines.open(data_args.data_path, "r") as reader:
        for idx, obj in enumerate(reader):
            question = obj["question"]
            candidates = obj["candidates"]
            cands_weight = weights[idx]
            assert len(candidates) == len(cands_weight)
            for i in range(len(candidates)):
                train_data.append({
                    "question": question,
                    "answer": candidates[i],
                    "weight": cands_weight[i]
                })
    train_dataset = SupervisedDataset(
        train_data,
        tokenizer=tokenizer,
        max_len=max_len,
        data_processor=lambda x: "Q: " + x["question"] + "\n" + "A: " + tokenizer.eos_token + x["answer"],
        weight_extractor=lambda x: x["weight"]
    )
    return train_dataset


def evaluation_main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # output_dir = "modelL-filter_strategy_{}-time_{}".format(data_args.data_filter_mode, int(time.time()))
    # training_args.output_dir = os.path.join(training_args.output_dir, output_dir)
    # os.makedirs(training_args.output_dir, exist_ok=True)
    ROLE_CONTENT = "You are a calculation assistant. You will be given an arithmetic question. Please think step by step and give the answer. After giving your thoughts, use 'The answer is:' followed by the answer."
    accelerator = Accelerator()
    device = accelerator.device

    logger.info('Loading causal model...')
    modelL = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )
    if len(model_args.peft_model_path) > 0:
        logger.info("loading peft weights from{}".format(model_args.peft_model_path))
        modelL = PeftModel.from_pretrained(modelL, model_args.peft_model_path)
        modelL.merge_and_unload()
    tokenizerL = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        padding_side="left")
    tokenizerL.pad_token_id = tokenizerL.eos_token_id
    terminators = [
        tokenizerL.eos_token_id,
        tokenizerL.convert_tokens_to_ids("<|eot_id|>")
    ]
    test_dataset = load_test_data(tokenizerL, data_args, training_args).sources
    dp, ge, qe = get_extractors(data_args.dataset_name)
    questions = [qe(x) for x in test_dataset]
    prompts = [dp(x) for x in test_dataset]
    answers = [ge(x) for x in test_dataset]
    prompts = ["Question: " + x for x in questions]

    def prepare_prompts(prompts, tokenizer, batch_size=16):
        messages = [
            gsm8k_opencompass_prompt + "\n" + x + "\nLet's think step by step\nAnswer:"
            for x in prompts
        ]
        # inputs = [
        #     tokenizerL.apply_chat_template(
        #         m,
        #         add_generation_prompt=True,
        #         return_tensors="pt",
        #         tokenize=False
        #     )
        #     for m in messages
        # ]
        inputs = messages
        batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        batches_tok = []
        tokenizer.padding_side = "left"
        for prompt_batch in batches:
            batches_tok.append(
                tokenizer(
                    prompt_batch,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=training_args.model_max_length,
                    add_special_tokens=True).to(device)
            )
        # tokenizer.padding_side="right"
        return batches_tok

    modelL.eval()
    modelL.to(device)
    accelerator.wait_for_everyone()

    with accelerator.split_between_processes(prompts) as prompts:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = prepare_prompts(prompts, tokenizerL, batch_size=training_args.per_device_eval_batch_size)
        pbar = tqdm.tqdm(total=len(prompt_batches), disable=(not accelerator.is_local_main_process))

        for prompts_tokenized in prompt_batches:
            with torch.no_grad():
                outputs_tokenized = modelL.generate(
                    **prompts_tokenized,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizerL.eos_token_id,
                    # do_sample=True
                )

            # remove prompt from gen. tokens
            outputs_tokenized = [tok_out[len(tok_in):]
                                 for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]

            # count and decode gen. tokens
            num_tokens = sum([len(t) for t in outputs_tokenized])
            outputs = tokenizerL.batch_decode(outputs_tokenized)

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
            if accelerator.is_local_main_process:
                pbar.update(1)
            torch.cuda.empty_cache()
        results = [results]  # transform to list, otherwise gather_object() will not collect correctly
    results_gathered = gather_object(results)
    if accelerator.is_main_process:
        total_results = []
        for r in results_gathered:
            total_results += r["outputs"]
        total_results = [answer.split(tokenizerL.eos_token)[0] if tokenizerL.eos_token in answer else answer for answer
                         in total_results]

        if data_args.dataset_name in ["gsm8k", "ChilleD/SVAMP"]:
            pred_answers = [derive_num_from_output(re) for re in total_results]
        elif data_args.dataset_name in ["aqua_rat", "allenai/openbookqa", "facebook/anli", "facebook/anli2",
                                        "ChilleD/StrategyQA"]:
            pred_answers = [derive_choice_from_output(re) for re in total_results]
        else:
            logger.info("Invalid dataset name")
            quit()
        assert len(pred_answers) == len(answers)
        acc = 0
        for i in range(len(questions)):
            if data_args.dataset_name in ["gsm8k", "ChilleD/SVAMP"]:
                acc += 1 if (pred_answers[i] is not None and int(float(pred_answers[i])) == int(
                    float(answers[i]))) else 0
            elif data_args.dataset_name in ["aqua_rat", "allenai/openbookqa", "facebook/anli", "facebook/anli2",
                                            "ChilleD/StrategyQA"]:
                acc += 1 if (pred_answers[i] is not None and pred_answers[i] == answers[i]) else 0
        acc = acc / len(pred_answers)
        logger.info(f"acc is {acc}")
        # dump results
        dump_path = model_args.peft_model_path if len(model_args.peft_model_path) else "./"
        with open(os.path.join(dump_path, "debug_{}.txt".format(os.environ.get("SEED", 114514))), "w",
                  encoding="utf8") as f:
            for i in range(len(questions)):
                f.write(questions[i])
                f.write("\n")
                f.write("A: " + str(total_results[i]))
                f.write("\n")
                f.write("Ground: " + str(answers[i]))
                f.write("\n")
                f.write("-----------------------------")
                f.write("\n")
        with open(os.path.join(dump_path, "acc_{}.txt".format(os.environ.get("SEED", 114514))), "w",
                  encoding="utf8") as f:
            f.write(str(acc))


if __name__ == "__main__":
    # 注意seed，原设置是没有do_sample的
    seed = os.environ.get("SEED", 114514)
    seed = int(seed)
    print("================set global random seed to {}================".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    evaluation_main()


