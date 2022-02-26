import torch
import torch.distributions as dist

import random

class ObjectConcept:
    def __init__(
        self,
        shape,
        color,
        weight,
        roughness
    ):
        self.shape = shape # shape is a cateogirical dist
        self.color = color # color is a categorical dist
        self.weight = weight # weight is a normal dist
        self.roughness = roughness # roughness is a normal dist

    def sample(self, destroy_sensor=False, set_none_zero=True):
        data = [
            self.shape.sample(),
            self.color.sample(),
            self.weight.sample(),
            self.roughness.sample()
        ]
        if destroy_sensor:
            if set_none_zero:
                data[random.randint(0, 3)] = 0
            else:
                data[random.randint(0, 3)] = None
        return data

    def log_prob(self, x):
        log_prob = 0.0

        if x[0] != None:
            log_prob += self.shape.log_prob(x[0])
        if x[1] != None:
            log_prob += self.color.log_prob(x[1])
        if x[2] != None:
            log_prob += self.weight.log_prob(x[2])
        if x[3] != None:
            log_prob += self.roughness.log_prob(x[3])

        return log_prob
        
class Model:
    def __init__(self, object_concepts):
        self.object_concepts = object_concepts

    def sample(self, batch_size=16, destroy_sensor=False, set_none_zero=True):
        data = []
        labels = []
        for _ in range(batch_size):
            concept = random.choice(self.object_concepts)
            concept_idx = self.object_concepts.index(concept)
            data.append(
                concept.sample(destroy_sensor, set_none_zero)
            )
            labels.append(concept_idx)
        return data, labels

    def infer(self, x):
        winner_prob = -100
        winner_idx = -1
        for idx, concept in enumerate(self.object_concepts):
            log_prob = concept.log_prob(x)
            if log_prob > winner_prob:
                winner_prob = log_prob
                winner_idx = idx
        return winner_prob, winner_idx

if __name__ == "__main__":

    shape_label = ["round", "box", "cylinder"]
    color_label = ["red", "white", "green", "yellow"]

    football_shape = dist.Categorical(probs=torch.tensor([0.95, 0.025, 0.025]))
    football_color = dist.Categorical(probs=torch.tensor([0.1, 0.7, 0.05, 0.15]))
    football_weight = dist.Normal(torch.tensor(420.0), torch.tensor(10.0)) # in grams
    football_roughness = dist.Normal(torch.tensor(-5.0), torch.tensor(1.0))

    football = ObjectConcept(
        football_shape, football_color,
        football_weight, football_roughness
    )

    model = Model([football, football, football])

    data, labels = model.sample()
    for d in data:
        t = model.infer(d)
        print(torch.tensor(d))

    print(torch.tensor(data).shape)

    # print(shape_label[sample[0].item()], color_label[sample[1].item()])
    # print(football.log_prob(sample))  