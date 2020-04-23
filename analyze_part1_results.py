
import torch

model_load_path = "model1 enson.tar"

checkpoint = torch.load(model_load_path)

#for key in checkpoint:
#    print(key, checkpoint[key])


print( checkpoint["history"]["test_acc"][-1])



