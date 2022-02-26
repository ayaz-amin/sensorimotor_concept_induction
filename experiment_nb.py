# This script trains and tests a naive Bayes classifier (the probabilistic model referred to in the project)
# It is essentially the robot simulator, which is implemented as a probabilistic model,
# without without the ground-truth probabilities. Training this model brings the probabilities
# closer to the ground-truth probabilities of the simulator. This is done to account for the fact
# that ground-truth parameters are not available in the real world. Also, it would be cheating to 
# just use the hard-coded simulator, since it is essentially just winning at its own game

import torch
import torch.nn.functional as F
import torch.distributions as dist

from src.model import ObjectConcept, Model

NUM_ITERATIONS = 1000
BATCH_SIZE = 32
TEST_SIZE = 1000

def learn(data, labels):
    football_shape = torch.tensor([0.0, 0.0, 0.0])
    football_color = torch.tensor([0.0, 0.0, 0.0, 0.0])
    football_weight = []
    football_roughness = []

    computer_shape = torch.tensor([0.0, 0.0, 0.0])
    computer_color = torch.tensor([0.0, 0.0, 0.0, 0.0])
    computer_weight = [] # in grams
    computer_roughness = []

    bottle_shape = torch.tensor([0.0, 0.0, 0.0])
    bottle_color = torch.tensor([0.0, 0.0, 0.0, 0.0])
    bottle_weight = [] # in grams
    bottle_roughness = []

    fan_shape = torch.tensor([0.0, 0.0, 0.0])
    fan_color = torch.tensor([0.0, 0.0, 0.0, 0.0])
    fan_weight = [] # in grams
    fan_roughness = []
 
    for i, (x, y) in enumerate(list(zip(data, labels))):
        if y == 0:
            football_shape[x[0]] += 1
            football_color[x[1]] += 1
            football_weight.append(x[2])
            football_roughness.append(x[3])
        elif y == 1:
            computer_shape[x[0]] += 1
            computer_color[x[1]] += 1
            computer_weight.append(x[2])
            computer_roughness.append(x[3])
        elif y == 2:
            bottle_shape[x[0]] += 1
            bottle_color[x[1]] += 1
            bottle_weight.append(x[2])
            bottle_roughness.append(x[3])
        elif y == 3:
            fan_shape[x[0]] += 1
            fan_color[x[1]] += 1
            fan_weight.append(x[2])
            fan_roughness.append(x[3])

    football_shape = dist.Categorical(probs=F.softmax(football_shape, dim=0))
    football_color = dist.Categorical(probs=F.softmax(football_color, dim=0))
    football_weight = torch.std_mean(torch.tensor(football_weight), unbiased=True)
    football_weight = dist.Normal(football_weight[1], football_weight[0])
    football_roughness = torch.std_mean(torch.tensor(football_roughness), unbiased=True)
    football_roughness = dist.Normal(football_roughness[1], football_roughness[0])

    computer_shape = dist.Categorical(F.softmax(computer_shape, dim=0))
    computer_color = dist.Categorical(F.softmax(computer_color, dim=0))
    computer_weight = torch.std_mean(torch.tensor(computer_weight), unbiased=True)
    computer_weight = dist.Normal(computer_weight[1], computer_weight[0])
    computer_roughness = torch.std_mean(torch.tensor(computer_roughness), unbiased=True)
    computer_roughness = dist.Normal(computer_roughness[1], computer_roughness[0])

    bottle_shape = dist.Categorical(F.softmax(bottle_shape, dim=0))
    bottle_color = dist.Categorical(F.softmax(bottle_color, dim=0))
    bottle_weight = torch.std_mean(torch.tensor(bottle_weight), unbiased=True)
    bottle_weight = dist.Normal(bottle_weight[1], bottle_weight[0])
    bottle_roughness = torch.std_mean(torch.tensor(bottle_roughness), unbiased=True)
    bottle_roughness = dist.Normal(bottle_roughness[1], bottle_roughness[0])

    fan_shape = dist.Categorical(F.softmax(fan_shape, dim=0))
    fan_color = dist.Categorical(F.softmax(fan_color, dim=0))
    fan_weight = torch.std_mean(torch.tensor(fan_weight), unbiased=True)
    fan_weight = dist.Normal(fan_weight[1], fan_weight[0])
    fan_roughness = torch.std_mean(torch.tensor(fan_roughness), unbiased=True)
    fan_roughness = dist.Normal(fan_roughness[1], fan_roughness[0])

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

    return Model(
        [football, computer, bottle, fan]
    )


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

    generator = Model([football, computer, bottle, fan])
    train_data, train_labels = generator.sample(batch_size=NUM_ITERATIONS * BATCH_SIZE, destroy_sensor=False)
    model = learn(train_data, train_labels)
    sample_clean, labels_clean = generator.sample(batch_size=TEST_SIZE, destroy_sensor=False, set_none_zero=False)
    sample_corrupt, labels_corrupt = generator.sample(batch_size=TEST_SIZE, destroy_sensor=True, set_none_zero=False)
    
    sum_probs = 0.0
    correct_count = 0
    for i, s in enumerate(sample_clean):
        winner_prob, winner_idx = model.infer(s)
        if winner_idx == labels_clean[i]:
            correct_count += 1
        sum_probs += winner_prob

    print("Infernce performance on clean data")
    print("==================================")
    print(
        "Accuracy: {} %, Average Log Probability: {}".format(
            100 * correct_count / TEST_SIZE, 
            sum_probs / TEST_SIZE
        )
    )

    print("")

    sum_probs = 0.0
    correct_count = 0
    for i, s in enumerate(sample_corrupt):
        winner_prob, winner_idx = model.infer(s)
        if winner_idx == labels_corrupt[i]:
            correct_count += 1
        sum_probs += winner_prob

    print("Infernce performance on corrupt data")
    print("==================================")
    print(
        "Accuracy: {} %, Average Log Probability: {}".format(
            100 * correct_count / TEST_SIZE, 
            sum_probs / TEST_SIZE
        )
    )
