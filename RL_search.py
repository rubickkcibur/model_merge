import numpy as np
import torch
from utils import compute_metrics
import torch.nn as nn
import copy
import tqdm
import os
import json

class Environment():
    def __init__(self, args):
        # state: np.array, [0-1]*32
        # layer_names: list[list] inner list is the parameter names of each layer
        self.state = args["seed_state"]
        self.target_model = args["target_model"]
        self.peer_model = args["peer_model"]
        self.layer_names = args["layer_names"]
        self.tokenizer = args["tokenizer"]
        self.device = args["device"]
        self._init_baseline_reward()
        assert len(self.state) == len(self.layer_names)
        pass

    def _init_baseline_reward(self):
        self._merge_weight()
        self.baseline_r = compute_metrics(self.target_model, self.tokenizer, self.device)
        self._recover_weight()
        return
    
    def _merge_weight(self):
        with torch.no_grad():
            updated_layers = []
            for i in range(len(self.state)):
                layers = self.layer_names[i]
                weight = self.state[i]
                for name, p in self.target_model.named_parameters():
                    if name in layers:
                        p.data *= weight
                        p.data += self.peer_model.state_dict()[name] * (1 - weight)
                        updated_layers.append(name)

            for name, p in self.target_model.named_parameters():
                if name not in updated_layers:
                    p.data *= 0.5
                    p.data += self.peer_model.state_dict()[name] * 0.5
        return
    def _recover_weight(self):
        with torch.no_grad():
            updated_layers = []
            for i in range(len(self.state)):
                layers = self.layer_names[i]
                weight = self.state[i]
                for name, p in self.target_model.named_parameters():
                    if name in layers:
                        p.data -= self.peer_model.state_dict()[name] * (1 - weight)
                        p.data /= weight
                        updated_layers.append(name)

            for name, p in self.target_model.named_parameters():
                if name not in updated_layers:
                    p.data -= self.peer_model.state_dict()[name] * 0.5
                    p.data /= 0.5
        return

    def step(self, action):
        # action: np.array, [-1 - 1]*32
        assert len(self.state) == len(action)
        self.state += action
        self.state = np.clip(self.state, 0.01, 1)
        # print("state: ", self.state)
        return self.state
    
    def reward(self):
        self._merge_weight()
        r = compute_metrics(self.target_model, self.tokenizer, self.device)
        # print("reward:", r)
        # print("self.baseline_r:", self.baseline_r)
        self._recover_weight()
        return r - self.baseline_r
    
    def set_seed_state(self, seed_state):
        self.state = seed_state
        self._init_baseline_reward()
        return



class ReplayBuffer():
    def __init__(self, args):
        self.max_size = args["max_size"]
        self.buffer = []

    def push(self, s, a, r, n_s, end):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append({
            "state": s,
            "action": a,
            "reward": r,
            "n_state": n_s,
            "terminal": end
        })
        return
    
    def sample(self, batch_size):
        assert batch_size < len(self.buffer)
        idxs = list(np.random.choice(len(self.buffer), batch_size, replace=False))
        return [
            self.buffer[i]
            for i in idxs
        ]
    
    def empty(self):
        self.buffer = []
        return


class DDPG():
    def __init__(self, args):
        state_size = args["state_size"]
        action_size = args["action_size"]
        hidden_size = args["hidden_size"]
        actor_lr = args["actor_lr"]
        critic_lr = args["critic_lr"]
        gamma = args["gamma"]
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.ReLU()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )
        self._init_weights()
        self.update_target_network()
        self.optimizer = torch.optim.Adam([
            {"params": self.actor.parameters(), "lr": actor_lr},
            {"params": self.critic.parameters(), "lr": critic_lr},
        ])
        self.gamma = gamma
    
    def _init_weights(self):
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def train(self):
        self.actor.train()
        self.critic.train()

    def to_device(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)            

    def forward_a(self, state):
        # state is a tensor on device
        action_raw = self.actor(state)
        return action_raw
    
    def predict_target_a(self, state, no_grad=False):
        if no_grad:
            with torch.no_grad():
                action_raw = self.target_actor(state)
            return action_raw
        else:
            action_raw = self.target_actor(state)
            return action_raw
    
    def forward_q(self, state, action):
        input = torch.concat([state, action], dim = -1)
        q = self.critic(input)
        return q
    
    def predict_target_q(self, state, action, no_grad=False):
        if no_grad:
            with torch.no_grad():
                input = torch.concat([state, action], dim = -1)
                q = self.target_critic(input)
            return q
        else:
            input = torch.concat([state, action], dim = -1)
            q = self.target_critic(input)
            return q
    
    def update_actor(self, inputs):
        # inputs: dict, "state": state tensor
        state = inputs["state"]
        action_raw = self.forward_a(state)
        q = self.predict_target_q(state, action_raw)
        loss = -q
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        return loss.mean().detach().cpu().numpy()
    
    def update_critic(self, inputs):
        state = inputs["state"]
        action = inputs["action"]
        reward = inputs["reward"] # batch_sdize, 1
        n_state = inputs["n_state"]
        terminal = inputs["terminal"]
        n_action = self.predict_target_a(n_state).detach()
        q = self.forward_q(state, action)
        target_q = self.predict_target_q(n_state, n_action).detach()
        target_q = self.gamma * target_q * terminal + reward
        loss = 0.5*(q - target_q)**2
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        return loss.mean().detach().cpu().numpy()

    def update_target_network(self):
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor.eval()
        self.target_critic.eval()
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False
        return

    def save_model(self, actor_path, critic_path):
        torch.save(self.target_actor.state_dict(), actor_path)
        torch.save(self.target_critic.state_dict(), critic_path)
        return
    
    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.update_target_network()
        return


class LearningFramework():
    def __init__(self, args):
        self.env = Environment(args["env_args"])
        self.replay_buffer = ReplayBuffer(args["buffer_args"])
        self.policy = DDPG(args["policy_args"])
        self.layers_num = len(args["env_args"]["layer_names"])
        self.max_T = args["max_T"]
        self.device = args["device"]
        self.policy.to_device(self.device)
        self.replay_buffer.empty()
        self.epochs = args["epochs"]
        self.batch_size = args["batch_size"]
        self.actor_path = args["actor_path"]
        self.critic_path = args["critic_path"]
        self.log_path = args["log_path"]

    def collect_data(self, epochs):
        print("collecting data......")
        with tqdm.tqdm(total = epochs * self.max_T) as pbar:
            rewards = []
            for i in range(epochs):
                seed_state = np.random.rand(self.layers_num)
                self.env.set_seed_state(seed_state)
                state = seed_state
                avg_reward = 0
                for j in range(self.max_T):
                    input_state = torch.tensor(state).to(torch.float32).to(self.device)
                    input_state = input_state.unsqueeze(0)
                    pre_action = self.policy.predict_target_a(input_state, no_grad=True)
                    action = pre_action.squeeze().detach().cpu().numpy()
                    action_cliped = np.clip(action, 0, 1)
                    action_cliped = action_cliped - 0.5
                    action_cliped *= 0.2
                    n_state = self.env.step(action_cliped)
                    reward = self.env.reward()
                    terminal = 1 if (np.linalg.norm(n_state - state) < 1e-2) or (j == self.max_T - 1) else 0
                    self.replay_buffer.push(
                        state,
                        action,
                        reward,
                        n_state,
                        terminal
                    )
                    avg_reward += reward
                    if terminal > 0:
                        break
                    state = n_state.copy()
                    pbar.update(1)
                rewards.append(avg_reward/(j+1))
        return rewards

    def train(self, steps):
        print("training network.....")
        actor_losses = []
        critic_losses = []
        with tqdm.tqdm(total=steps) as pbar:
            for i in range(steps):
                inputs = self.replay_buffer.sample(self.batch_size)
                state = np.array([d["state"] for d in inputs])
                action = np.array([d["action"] for d in inputs])
                n_state = np.array([d["n_state"] for d in inputs])
                reward = np.array([d["reward"] for d in inputs]).reshape(-1, 1)
                terminal = np.array([d["terminal"] for d in inputs]).reshape(-1, 1)
                input_tensor = {
                    "state": torch.tensor(state).to(torch.float32).to(self.device),
                    "action": torch.tensor(action).to(torch.float32).to(self.device),
                    "n_state": torch.tensor(n_state).to(torch.float32).to(self.device),
                    "reward": torch.tensor(reward).to(torch.float32).to(self.device),
                    "terminal": torch.tensor(terminal).to(torch.float32).to(self.device)
                }
                self.policy.train()
                actor_loss = self.policy.update_actor(input_tensor)
                critic_loss = self.policy.update_critic(input_tensor)
                actor_losses.append(float(actor_loss))
                critic_losses.append(float(critic_loss))
                pbar.update(1)
        return actor_losses, critic_losses

    def learn(self):
        actor_losses = []
        critic_losses = []
        rewards = []
        for i in range(self.epochs):
            rewards += self.collect_data(60)
            a_loss, c_loss = self.train(100)
            actor_losses += a_loss
            critic_losses += c_loss
            self.policy.update_target_network()
        self.policy.save_model(self.actor_path, self.critic_path)
        with open(os.path.join(self.log_path, "loss.json"), "w") as f:
            json.dump({
                "actor_loss": actor_losses,
                "critic_loss": critic_losses
            }, f)
        with open(os.path.join(self.log_path, "rewards.json"), "w") as f:
            json.dump({
                "rewards": rewards,
            }, f)

    def inference(self, seed_state, max_steps):
        states = []
        state = seed_state
        self.env.set_seed_state(seed_state)
        for i in range(max_steps):
            input_state = torch.tensor(state).to(torch.float32).to(self.device)
            input_state = input_state.unsqueeze(0)
            pre_action = self.policy.predict_target_a(input_state, no_grad=True)
            action = pre_action.squeeze().detach().cpu().numpy()
            action_cliped = np.clip(action, 0, 1)
            action_cliped = action_cliped - 0.5
            action_cliped += 0.2
            n_state = self.env.step(action_cliped)
            states.append(n_state)
            state = n_state
        return states
    
    def load_policy(self, actor_path, critic_path):
        self.policy.load_model(actor_path, critic_path)
        return
        