import matplotlib.pyplot as plt
import numpy as np
import json
with open("/home/rubickjiang/model_merge/RL/logs/rewards.json", "r") as f:
    rewards = json.load(f)
rewards = rewards["rewards"]
x = list(range(len(rewards)))
plt.plot(x, rewards)
plt.savefig("./rewards.png")