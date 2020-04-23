from FaceRecognizerModel import *
from network_operator import ModelOperator
from DataLoader import *
import torch
from torch.nn import CrossEntropyLoss
from torch import optim
import matplotlib.pyplot as plt

MODEL_SAVE_PATH = "model1_part2.tar"
LR = 1e-3
Momentum = 0.9 # If you use SGD with momentum
BATCH_SIZE = 16
USE_CUDA = True
NUM_EPOCHS = 1
PATIENCE = 5
ALEXNET_EXTRACTOR = False
FINETUNE_ALEXNET = False
VISUALIZE = True

if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True

    train_loader, val_loader = get_train_data_loader(batch_size=BATCH_SIZE,
                                                     is_alexnet_used=ALEXNET_EXTRACTOR, validate=True)

    number_of_class = len(train_loader.dataset.classes)

    if FINETUNE_ALEXNET:
        model = FaceRecognizerAlexNetModel(class_number=number_of_class, use_cuda=USE_CUDA)
        ALEXNET_EXTRACTOR = True    # otomatik true olmasi lazim cunku data augmentation buna gore yapiliyor
        #optimizer = optim.Adam(model.get_learning_params(), lr=LR)
    else:
        model = FaceRecognizerModel(class_number=number_of_class, use_cuda=USE_CUDA,
                                    use_alexnet_extractor=ALEXNET_EXTRACTOR)

    if VISUALIZE:
        model = FaceRecognizerVisualizeModel(class_number=number_of_class, use_cuda=USE_CUDA,
                                             use_alexnet_extractor=ALEXNET_EXTRACTOR)

    optimizer = optim.Adam(model.get_learning_params(), lr=LR)
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

    test_loader = get_test_data_loader(batch_size=BATCH_SIZE, is_alexnet_used=ALEXNET_EXTRACTOR)[0]
    operator.test(test_loader)

    # Visualize feature maps
    ###################################################
    ##### Layer ciktilari mi yoksa layer weigtleri mi gosterilecek? Su anda layer weigtleri gosteriliyor
    # activations = {}
    # model.hidden_fc_layer_2.register_forward_hook(get_activation('hidden_fc_layer_2'))

    data, _ = val_loader.dataset[0]
    data.unsqueeze_(0)

    model.eval()
    output = model(data.cuda())

    """"
    act = activations['hidden_fc_layer_2']
    plt.imshow(act.cpu().numpy()) # matshow
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    """""
    plt.matshow(model.hidden_fc_layer_2.weight.data.cpu().numpy())  # matshow ve imshow
    plt.show()

    ##############################################333

