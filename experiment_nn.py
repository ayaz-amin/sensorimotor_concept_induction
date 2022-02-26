# This script trains and tests a single-layer neural network on simulated data.
# The neural network is used as a benchmark for the naive Bayes model. I created a single-layer neural network
# in accordance to the fact that a single-layer neural network (or more specifically, a linear model) is the 
# discriminative analogue to the generative naive Bayes model. This puts both models on equal ground.
# Keep in mind that training this model is unstable, it may require several runs to attain good results.
# This script additionally plots and saves a graph of the training error with respect to each iteration.

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt

from src.model import ObjectConcept, Model

NUM_ITERATIONS = 1000
BATCH_SIZE = 32
TEST_SIZE = 1000

if __name__ == "__main__":
    football_shape = dist.Categorical(probs=torch.tensor([0.95, 0.025, 0.025]))
    football_color = dist.Categorical(probs=torch.tensor([0.1, 0.7, 0.05, 0.15]))
    football_weight = dist.Normal(torch.tensor(420.0), torch.tensor(10.0)) # in grams
    football_roughness = dist.Normal(torch.tensor(-5.0), torch.tensor(1.0))

    computer_shape = dist.Categorical(probs=torch.tensor([0.025, 0.95, 0.025]))
    computer_color = dist.Categorical(probs=torch.tensor([0.15, 0.6, 0.15, 0.1]))
    computer_weight = dist.Normal(torch.tensor(8600.0), torch.tensor(100.0)) # in grams
    computer_roughness = dist.Normal(torch.tensor(-3.0), torch.tensor(1.0))

    bottle_shape = dist.Categorical(probs=torch.tensor([0.01, 0.85, 0.14]))
    bottle_color = dist.Categorical(probs=torch.tensor([0.01, 0.75, 0.01, 0.23]))
    bottle_weight = dist.Normal(torch.tensor(19.0), torch.tensor(5.0))
    bottle_roughness = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

    fan_shape = dist.Categorical(probs=torch.tensor([0.01, 0.95, 0.04]))
    fan_color = dist.Categorical(probs=torch.tensor([0.01, 0.95, 0.01, 0.2]))
    fan_weight = dist.Normal(torch.tensor(4500.0), torch.tensor(500.0))
    fan_roughness = dist.Normal(torch.tensor(1.0), torch.tensor(1.0))

    football = ObjectConcept(
        football_shape, football_color,
        football_weight, football_roughness
    )

    computer = ObjectConcept(
        computer_shape, computer_color, 
        computer_weight, computer_roughness
    )

    bottle = ObjectConcept(
        bottle_shape, bottle_color,
        bottle_weight, bottle_roughness
    )

    fan = ObjectConcept(
        fan_shape, fan_color,
        fan_weight, fan_roughness
    )

    model = Model([football, computer, bottle, fan])
    network = nn.Sequential(nn.Linear(4, 4), nn.LogSoftmax())
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    network.train()

    iters = []
    losses = []
    for i in range(NUM_ITERATIONS):
        optimizer.zero_grad()
        data, labels = model.sample(batch_size=BATCH_SIZE)
        data, labels = torch.tensor(data), torch.tensor(labels)
        y_pred = network(data)
        loss = F.cross_entropy(y_pred, labels, reduction="mean")
        loss.backward()
        optimizer.step()
        iters.append(i+1)
        losses.append(loss.item())

    plt.xlabel("Iterations")
    plt.ylabel("Loss (Cross Entropy)")
    plt.plot(np.array(iters), np.array(losses))
    plt.savefig("neural_network_training.png")
    
    sample_clean, labels_clean = model.sample(batch_size=TEST_SIZE, destroy_sensor=False)
    sample_corrupt, labels_corrupt = model.sample(batch_size=TEST_SIZE, destroy_sensor=True)
    
    correct_count = 0
    y_pred = network(torch.tensor(sample_clean))
    for i, pred in enumerate(y_pred):
        if torch.argmax(pred) == labels_clean[i]:
            correct_count += 1

    print("Infernce performance on clean data")
    print("==================================")
    print("Accuracy: {} %".format(100 * correct_count / TEST_SIZE))

    print("")

    correct_count = 0
    y_pred = network(torch.tensor(sample_corrupt))
    for i, pred in enumerate(y_pred):
        if torch.argmax(pred) == labels_corrupt[i]:
            correct_count += 1

    print("Infernce performance on corrupt data")
    print("==================================")
    print("Accuracy: {} %".format(100 * correct_count / TEST_SIZE))