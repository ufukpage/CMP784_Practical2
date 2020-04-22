import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# todo odevi gonderirken bu demoyu silicez


activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model = nn.Sequential(
    nn.Linear(10, 25),
    nn.ReLU(),
    nn.Linear(25, 2)
)

weights = model[0].weight.data.numpy()
model[0].register_forward_hook(get_activation('layer0'))

x = torch.randn(1, 10)
output = model(x)

plt.matshow(activations['layer0'])
plt.matshow(weights)
plt.show()