# This script trains and tests an XGBoost model on simulated data. This has been trained as a control
# since the naive Bayes was getting 100% accuracy on the dataset (even though it did not have the ground-truth probabilities).
# To confirm that there was no bias in the dataset that was giving the naive Bayes such high accuracies, this
# model has been trained to ensure that it was getting high accuracies on the data as well. Running this 
# script confirms that it XGBoost can also attain 100% accuracy (at least on my machine) on the dataset, 
# so I can assert there was no inherent bias in the simulator. 

import numpy as np

import torch
import torch.distributions as dist
import xgboost as xgb

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

    generator = Model([football, computer, bottle, fan])

    data, labels = generator.sample(batch_size=BATCH_SIZE * NUM_ITERATIONS)
    data, labels = torch.tensor(data).numpy(), torch.tensor(labels).numpy()

    clf = xgb.XGBClassifier(max_depth=5, num_classes=4, use_label_encoder=False)
    clf.fit(data, labels)
    
    test_data, test_labels = generator.sample(batch_size=TEST_SIZE)
    test_data, test_labels = torch.tensor(test_data).numpy(), torch.tensor(test_labels).numpy()
    
    corrupt_data, corrupt_labels = generator.sample(batch_size=TEST_SIZE, destroy_sensor=True)
    corrupt_data, corrupt_labels = torch.tensor(corrupt_data).numpy(), torch.tensor(corrupt_labels).numpy()

    test_probs = clf.predict_proba(test_data)
    corrupt_probs = clf.predict_proba(corrupt_data)

    correct = 0
    for i, p in enumerate(test_probs):
        if(np.argmax(p, axis=0) == test_labels[i]):
            correct = correct + 1
    
    print("Infernce performance on clean data")
    print("==================================")
    print("Accuracy: {} %".format(100 * correct / TEST_SIZE))

    print("")

    correct = 0
    for i, p in enumerate(corrupt_probs):
        if(np.argmax(p, axis=0) == test_labels[i]):
            correct = correct + 1

    print("Infernce performance on corrupt data")
    print("==================================")
    print("Accuracy: {} %".format(100 * correct / TEST_SIZE))