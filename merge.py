import os.path
import torch
import transformers
import copy
import numpy as np
import random
from datasets import load_dataset
import re
import json
from utils import compute_metrics
from RL_search import LearningFramework

SPECIAL_MODEL_PATH = "/aifs4su/zhanqimin/LLaMA-Factory/merge_lora/"
code_base_model_path = os.path.join(SPECIAL_MODEL_PATH, "llama3_base_codealpaca_3epoch")
math_base_model_path = os.path.join(SPECIAL_MODEL_PATH, "llama3_base_metamath_3epoch")
chat_base_model_path = os.path.join(SPECIAL_MODEL_PATH, "llama3_base_ultra_chat_1epoch")

def format_ground_truth_answer(ground: str):
    ground = ground.replace("####", "The answer is")
    ground = re.sub(r"<<.*?>>","",ground)
    return ground


def merge_weight(target_model, peer_model, layer_weights):
    with torch.no_grad():
        for name, p in target_model.named_parameters():
            if name in layer_weights:
                p.data *= layer_weights[name]
                p.data += peer_model.state_dict()[name] * (1 - layer_weights[name])
            else:
                p.data *= 0.5
                p.data += peer_model.state_dict()[name] * 0.5
    return

def recover_weight(target_model, peer_model, layer_weights):
    with torch.no_grad():
        for name, p in target_model.named_parameters():
            if name in layer_weights:
                p.data -= peer_model.state_dict()[name] * (1 - layer_weights[name])
                p.data /= layer_weights[name]
            else:
                p.data -= peer_model.state_dict()[name] * 0.5
                p.data /= 0.5
    return


def find_optimal_weights(target_model, peer_model, tokenizer, LAYER_PARTS, WEIGHT_LIST):

    layer_weights = {}
    print(compute_metrics(target_model, tokenizer))
    print(compute_metrics(peer_model, tokenizer))

    search_records = []
    layers_per_part = 32 // LAYER_PARTS
    for part in range(LAYER_PARTS-1, -1, -1):
        layers = []
        for layer_n in range(part * layers_per_part, (part+1) * layers_per_part):
            layers.extend([
                "model.layers.{}.self_attn.q_proj.weight".format(layer_n),
                "model.layers.{}.self_attn.k_proj.weight".format(layer_n),
                "model.layers.{}.self_attn.v_proj.weight".format(layer_n),
                "model.layers.{}.self_attn.o_proj.weight".format(layer_n),
                "model.layers.{}.mlp.gate_proj.weight".format(layer_n),
                "model.layers.{}.mlp.up_proj.weight".format(layer_n),
                "model.layers.{}.mlp.down_proj.weight".format(layer_n),
                "model.layers.{}.input_layernorm.weight".format(layer_n),
                "model.layers.{}.post_attention_layernorm.weight".format(layer_n),
            ])
        loss_record = []
        for weight in WEIGHT_LIST:
            for name in layers:
                layer_weights[name] = weight
            merge_weight(target_model, peer_model, layer_weights)
            loss = compute_metrics(target_model, tokenizer)
            loss_record.append(float(loss))
            recover_weight(target_model, peer_model, layer_weights)
        search_records.append(loss_record)
        optimal_weight = WEIGHT_LIST[np.argmin(np.array(loss_record))]
        for name in layers:
            layer_weights[name] = optimal_weight
        print(loss_record)
    print(search_records)
    with open("optimal_weights_32parts_inverse_1000sample.json", "w") as f:
        json.dump(layer_weights, f)

def merge_models(target_model, peer_model, weight_file_path, save_path):
    with open(weight_file_path, "r") as f:
        layer_weights = json.load(f)
    with torch.no_grad():
        for name, p in target_model.named_parameters():
            if name in layer_weights:
                p.data *= layer_weights[name]
                p.data += peer_model.state_dict()[name] * (1 - layer_weights[name])
            else:
                p.data *= 0.5
                p.data += peer_model.state_dict()[name] * 0.5
    target_model.save_pretrained(save_path)

def learn_by_RL(model_math, model_code, tokenizer, layer_num):
    layer_names = [
        [
            "model.layers.{}.self_attn.q_proj.weight".format(layer_n),
            "model.layers.{}.self_attn.k_proj.weight".format(layer_n),
            "model.layers.{}.self_attn.v_proj.weight".format(layer_n),
            "model.layers.{}.self_attn.o_proj.weight".format(layer_n),
            "model.layers.{}.mlp.gate_proj.weight".format(layer_n),
            "model.layers.{}.mlp.up_proj.weight".format(layer_n),
            "model.layers.{}.mlp.down_proj.weight".format(layer_n),
            "model.layers.{}.input_layernorm.weight".format(layer_n),
            "model.layers.{}.post_attention_layernorm.weight".format(layer_n),
        ] for layer_n in range(layer_num)
    ]
    env_args = {
        "seed_state": np.random.rand(32),
        "target_model": model_math,
        "peer_model": model_code,
        "layer_names": layer_names,
        "tokenizer": tokenizer,
        "device": "cuda:5",
    }
    buffer_args = {
        "max_size": 2000
    }
    policy_args = {
        "state_size": layer_num,
        "action_size": layer_num,
        "hidden_size": 128,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "gamma": 0.8
    }
    args = {
        "env_args": env_args,
        "buffer_args": buffer_args,
        "policy_args": policy_args,
        "max_T": 10,
        "epochs": 20,
        "device": "cuda:5",
        "batch_size": 32,
        "actor_path": "/home/rubickjiang/model_merge/RL/actor.pth",
        "critic_path": "/home/rubickjiang/model_merge/RL/critic.pth",
        "log_path": "/home/rubickjiang/model_merge/RL/logs"
    }
    lr = LearningFramework(args)
    lr.learn()

def find_weight_by_RL(model_math, model_code, tokenizer, layer_num, weight_path):
    layer_names = [
        [
            "model.layers.{}.self_attn.q_proj.weight".format(layer_n),
            "model.layers.{}.self_attn.k_proj.weight".format(layer_n),
            "model.layers.{}.self_attn.v_proj.weight".format(layer_n),
            "model.layers.{}.self_attn.o_proj.weight".format(layer_n),
            "model.layers.{}.mlp.gate_proj.weight".format(layer_n),
            "model.layers.{}.mlp.up_proj.weight".format(layer_n),
            "model.layers.{}.mlp.down_proj.weight".format(layer_n),
            "model.layers.{}.input_layernorm.weight".format(layer_n),
            "model.layers.{}.post_attention_layernorm.weight".format(layer_n),
        ] for layer_n in range(layer_num)
    ]
    env_args = {
        "seed_state": np.random.rand(32),
        "target_model": model_math,
        "peer_model": model_code,
        "layer_names": layer_names,
        "tokenizer": tokenizer,
        "device": "cuda:5",
    }
    buffer_args = {
        "max_size": 2000
    }
    policy_args = {
        "state_size": layer_num,
        "action_size": layer_num,
        "hidden_size": 128,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "gamma": 0.8
    }
    args = {
        "env_args": env_args,
        "buffer_args": buffer_args,
        "policy_args": policy_args,
        "max_T": 10,
        "epochs": 20,
        "device": "cuda:5",
        "batch_size": 32,
        "actor_path": "/home/rubickjiang/model_merge/RL/actor.pth",
        "critic_path": "/home/rubickjiang/model_merge/RL/critic.pth",
        "log_path": "/home/rubickjiang/model_merge/RL/logs"
    }
    lr = LearningFramework(args)
    lr.load_policy(
        actor_path = "/home/rubickjiang/model_merge/RL/actor.pth",
        critic_path = "/home/rubickjiang/model_merge/RL/critic.pth" 
    )
    with open(weight_path, "r") as f:
        layer_weights = json.load(f)
    seed_weight = [0.5] * layer_num
    for i in range(len(layer_names)):
        layer_name = layer_names[i][0]
        weight = layer_weights[layer_name]
        seed_weight[i] = weight
    pred_states = lr.inference(np.array(seed_weight), 10)
    print(pred_states)


    

if __name__ == "__main__":
    seed = 114514
    LAYER_PARTS = 32
    WEIGHT_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]
    assert 32 % LAYER_PARTS == 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model_code = transformers.AutoModelForCausalLM.from_pretrained(
        code_base_model_path,
        torch_dtype=torch.bfloat16
    )
    model_math = transformers.AutoModelForCausalLM.from_pretrained(
        math_base_model_path,
        torch_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        math_base_model_path,
        model_max_length=2048,
        padding_side="left"
    )
    model_math.to("cuda:5")
    model_code.to("cuda:5")
    learn_by_RL(model_math, model_code, tokenizer, LAYER_PARTS)
    # find_weight_by_RL(model_math, model_code, tokenizer, LAYER_PARTS, "/home/rubickjiang/model_merge/optimal_weights_32parts_inverse.json")
    # find_optimal_weights(model_math, model_code, tokenizer, LAYER_PARTS, WEIGHT_LIST)
    # merge_models(model_math, model_code, "/home/rubickjiang/model_merge/optimal_weights_32parts_inverse_1000sample.json", "/aifs4su/rubickjiang/merge_models/32parts_inverse_1000sample")

# with torch.no_grad():
#     for name, p in model_math.named_parameters():
#         p.data *= 0.5
#         p.data += model_code.state_dict()[name] * 0.5
#
# model_math.save_pretrained("/aifs4su/rubickjiang/merge_models")

