import torch, os, sys
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


class LoaderClass(Dataset):
    def __init__(self, data, labels, phase, transforms):
        super(LoaderClass, self).__init__()
        self.transforms = transforms
        self.labels = labels[phase + "_labels"]
        self.data = data[phase + "_data"]
        self.phase = phase

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.data[idx]
        img = Image.fromarray(img)
        img = self.transforms(img)
        return img, torch.from_numpy(label)


def imagenet_transform(mode):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), #test zamaninda batch sayisini otamatik ayarlayabilmek icin aslinda arttirmis oluyoruz #todo bu duzeltilebilir
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms[mode]


""""
'train': transforms.Compose([
    transforms.Resize(im_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
"""
def simple_image_transform(mode):

    im_size = (28, 28)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(im_size),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(im_size),
            # transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    return data_transforms[mode]


def __get_data_loader(phases, batch_size, is_alexnet_used=False):
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    root_dataset_path = os.path.join(current_path, "dataset_part2")

    if is_alexnet_used:
        image_datasets = {x: datasets.ImageFolder(os.path.join(root_dataset_path, x), imagenet_transform(x))
                          for x in phases}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(root_dataset_path, x), simple_image_transform(x))
                          for x in phases}

    dataloaders = [torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                    shuffle=True, num_workers=4, pin_memory=True)
                   for x in phases]

    # dataset_sizes = {x: len(image_datasets[x]) for x in phases}
    # class_names = image_datasets['train'].classes

    return dataloaders


def get_train_data_loader(batch_size=16, validate=True, is_alexnet_used=False):
    if validate:
        phases = ['train', 'val']
    else:
        phases = ['train']
    return __get_data_loader(phases, batch_size, is_alexnet_used)


def get_train_test_data_loader(batch_size=16, validate=True, is_alexnet_used=False):
    if validate:
        phases = ['train', "test", 'val']
    else:
        phases = ['train', "test"]
    return __get_data_loader(phases, batch_size, is_alexnet_used)


def get_test_data_loader(batch_size=16, is_alexnet_used=False):
    return __get_data_loader(['test'], batch_size, is_alexnet_used)
