from FaceRecognizerModel import *
from network_operator import ModelOperator
from DataLoader import *
import torch
from torch.nn import CrossEntropyLoss
from torch import optim
import matplotlib.pyplot as plt


def run(model_save_path = "model2.tar", LR=1e-2, momentum=0.9, batch_size=512, alex_net_extractor = False,
        fine_tune_alexnet = True, visualize=True,
        num_epochs=200, patience=-1, scheduler_step=10, model_load_path=None,
        USE_CUDA=True):

    # torch.backends.cudnn.benchmark = True

    train_loader, val_loader = get_train_data_loader(batch_size=batch_size,
                                                     is_alexnet_used=alex_net_extractor, validate=True)

    number_of_class = len(train_loader.dataset.classes)

    if fine_tune_alexnet:
        model = FaceRecognizerAlexNetModel(class_number=number_of_class, use_cuda=USE_CUDA)
        alex_net_extractor = True    # otomatik true olmasi lazim cunku data augmentation buna gore yapiliyor
    else:
        model = FaceRecognizerModel(class_number=number_of_class, use_cuda=USE_CUDA,
                                    use_alexnet_extractor=alex_net_extractor)
        model.apply(init_linear_weights)

    if visualize:
        model = FaceRecognizerVisualizeModel(class_number=number_of_class, use_cuda=USE_CUDA,
                                             use_alexnet_extractor=alex_net_extractor)
        model.apply(init_linear_weights)

    optimizer = optim.Adam(model.get_learning_params(), lr=LR)
    criterion = CrossEntropyLoss()
    operator = ModelOperator(model, criterion, train_loader, val_loader, optimizer, num_epoch=num_epochs,
                             patience=patience, batch_size=batch_size, lr_scheduler=None)
    best_model = operator.train()
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': best_model.state_dict(),
        'model': model,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'history': operator.get_training_history(),
        #'transform': get_transform(),
        'patience': patience,
        'learning_rate': LR
    }, model_save_path
    )

    if model_load_path is not None:
        model_state = torch.load(model_save_path)
        # todo analiz kismi yazilacak

    test_loader = get_test_data_loader(batch_size=batch_size, is_alexnet_used=alex_net_extractor)[0]
    operator.test(test_loader)

    # Visualize feature maps
    ###################################################
    ##### Layer ciktilari mi yoksa layer weigtleri mi gosterilecek? Su anda layer weigtleri gosteriliyor
    # activations = {}
    # model.hidden_fc_layer_2.register_forward_hook(get_activation('hidden_fc_layer_2'))
    if visualize:
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

