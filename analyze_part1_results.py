
import torch

model_load_path = "model1_pooling_2_1_dropout_input_50.tar"

checkpoint = torch.load(model_load_path)

for key in checkpoint:
    print(key, checkpoint[key])



