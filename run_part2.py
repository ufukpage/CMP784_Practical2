from FaceRecognizerModel import *
from network_operator import ModelOperator
from DataLoader import *
import torch
from torch.nn import CrossEntropyLoss
from torch import optim


def run(model_save_path="model2.tar", LR=1e-2, momentum=0.9, batch_size=512, alex_net_extractor=False,
        fine_tune_alexnet = True, visualize=True,
        num_epochs=200, patience=-1, scheduler_step=10, model_load_path=None, weight_decay=0,
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

    optimizer = optim.Adam(model.get_learning_params(), lr=LR, weight_decay=weight_decay)

    criterion = CrossEntropyLoss()
    if scheduler_step > 0:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step)
    else:
        scheduler = None
    operator = ModelOperator(model, criterion, train_loader, val_loader, optimizer, num_epoch=num_epochs,
                             patience=patience, batch_size=batch_size, lr_scheduler=scheduler)
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
        'learning_rate': LR,
        'weight_decay': weight_decay,
        'scheduler_step': scheduler_step
    }, model_save_path
    )

    test_loader = get_test_data_loader(batch_size=batch_size, is_alexnet_used=alex_net_extractor)[0]
    operator.test(test_loader)

