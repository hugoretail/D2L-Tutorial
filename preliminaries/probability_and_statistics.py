import matplotlib.pyplot as plt
import random
import torch
from torch.distributions.multinomial import Multinomial

num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range (num_tosses)])
tails = num_tosses - heads
# print("heads, tails", [heads, tails])

fair_probs = torch.tensor([0.5,0.5])
Multinomial(100, fair_probs).sample()
# print(Multinomial(100, fair_probs).sample() / 100)

counts = Multinomial(1, fair_probs).sample((10000,))
# print(counts/10000)

cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)
estimates = estimates.detach().numpy()
plt.figure(figsize=(4.5, 3.5))
plt.plot(estimates[:, 0], label=("P(coin=heads)"))
plt.plot(estimates[:, 1], label=("P(coin=tails)"))
plt.axhline(y=0.5, color='black', linestyle='dashed')
plt.gca().set_xlabel('Samples')
plt.gca().set_ylabel('Estimated probability')
plt.legend()