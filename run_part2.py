from FaceRecognizerModel import *
from network_operator import ModelOperator
from DataLoader import *
import torch
from torch.nn import CrossEntropyLoss
from torch import optim
import matplotlib.pyplot as plt

MODEL_SAVE_PATH = "model1_part2.tar"
LR = 1e-4
Momentum = 0.9 # If you use SGD with momentum
BATCH_SIZE = 16
USE_CUDA = True
NUM_EPOCHS = 1
PATIENCE = -1
ALEXNET_EXTRACTOR = True        #Alexnet kullaninca bunun true olmasi gerekli transformationlar ona gore oluyor argumanlari ayarlarken otomatik bunu dusun

if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True

    train_loader, val_loader = get_train_data_loader(batch_size=BATCH_SIZE,
                                                     is_alexnet_used=ALEXNET_EXTRACTOR, validate=True)

    # model = FaceRecognizerModel(class_number=len(train_loader.dataset), use_cuda=USE_CUDA,
    #                            use_alexnet_extractor=ALEXNET_EXTRACTOR)

    model = FaceRecognizerAlexNetModel(class_number=len(train_loader.dataset), use_cuda=USE_CUDA)


    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    operator = ModelOperator(model, criterion, train_loader, val_loader, optimizer, num_epoch=NUM_EPOCHS,
                             patience=PATIENCE, batch_size=BATCH_SIZE, lr_scheduler=None)
    best_model = operator.train()
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion
    }, MODEL_SAVE_PATH
    )

    test_loader = get_test_data_loader(batch_size=1, is_alexnet_used=ALEXNET_EXTRACTOR)[0] # alexnet finetuninde batch size 1 olmali
    operator.test(test_loader)

    # Visualize feature maps

    """""
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    activations = {}
    model.hidden_fc_layer.register_forward_hook(get_activation('hidden_fc_layer'))
    data, _ = test_loader.dataset[0]
    data.unsqueeze_(0)
    output = model(data.cuda())
    act = activations['hidden_fc_layer']
    #fig, axarr = plt.subplots(act.size(0))
    #for idx in range(act.size(0)):
    plt.matshow(act.cpu().numpy())
    plt.show()
    """""

    ##############################################333
    # plt.matshow(act.cpu())

    # fig, axarr = plt.subplots(act.size(0))
    # for idx in range(act.size(0)):
    #     axarr[idx].imshow(act[idx].squeeze().cpu().numpy())


    #fig, axarr = plt.subplots(act.size(0))

    #plt.matshow(activations['hidden_fc_layer'].cpu())
