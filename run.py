from CnnModel import SimpleCNN
from network_operator import ModelOperator
from DataLoader import LoaderClass
import torch
from torch.nn import CrossEntropyLoss
from torch import optim
from torchvision import transforms
import numpy as np
import os
from PIL import Image, ImageFile


NUM_ARTISTS = 11
USE_CUDA = True
ImageFile.LOAD_TRUNCATED_IMAGES = True # Do not change this

def load_artist_data():
    data = []
    labels = []
    artists = [x for x in os.listdir(DATA_PATH) if x != '.DS_Store']
    print(artists)
    for folder in os.listdir(DATA_PATH):
        class_index = artists.index(folder)
        for image_name in os.listdir(DATA_PATH + "/" + folder):
            img = Image.open(DATA_PATH + "/" + folder + "/" + image_name)
            artist_label = (np.arange(NUM_ARTISTS) == class_index).astype(np.float32)
            data.append(np.array(img))
            labels.append(artist_label)
    shuffler = np.random.permutation(len(labels))
    data = np.array(data)[shuffler]
    labels = np.array(labels)[shuffler]

    length = len(data)
    val_size = int(length*0.2)
    val_data = data[0:val_size+1]
    train_data = data[val_size+1::]
    val_labels = labels[0:val_size+1]
    train_labels = labels[val_size+1::]
    print(val_labels)
    data_dict = {"train_data": train_data,"val_data": val_data}
    label_dict = {"train_labels": np.array(train_labels), "val_labels": np.array(val_labels)}

    return data_dict, label_dict


def get_transform():
    t = {
        'train': transforms.Compose([
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
    return t


def run(model_save_path, LR, momentum, batch_size, pooling,
        num_epochs, patience, path, scheduler_step=10, model_load_path=None):

    global DATA_PATH
    DATA_PATH = path
    # torch.backends.cudnn.benchmark = True
    data,labels = load_artist_data()
    model = SimpleCNN(use_cuda=USE_CUDA, pooling=pooling)
    # optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=5e-3)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    transforms = get_transform()
    train_dataset = LoaderClass(data,labels,"train",transforms["train"])
    valid_dataset = LoaderClass(data,labels,"val",transforms["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=4, pin_memory=True)

    criterion = CrossEntropyLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step)
    operator = ModelOperator(model, criterion, train_loader, val_loader, optimizer, num_epoch=num_epochs, patience=patience,
                              batch_size=batch_size, lr_scheduler=None)
    best_model = operator.train()

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': best_model.state_dict(),
        'model': model,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'history': operator.get_training_history(),
        'transform': get_transform(),
        'pooling': pooling,
        'patience': patience,
        'learning_rate': LR
    }, model_save_path
    )

    if model_load_path is not None:
        model_state = torch.load(model_save_path)
        # todo analiz kismi yazilacak