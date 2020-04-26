import torch
import torch.nn as nn
from torchvision import models


# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class AlexNetConv4(nn.Module):
    def __init__(self):
        super(AlexNetConv4, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = nn.Sequential(
            *list(original_model.features.children())[:-3]
        )

    def forward(self, x):
        x = self.features(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def init_linear_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


class FaceRecognizerModel(nn.Module):

    def __init__(self, class_number, use_cuda=True, use_alexnet_extractor=False):
        super(FaceRecognizerModel, self).__init__()
        self.use_alexnet_extractor = use_alexnet_extractor
        self.use_cuda = use_cuda
        self.feature_extractor = None


        if use_alexnet_extractor:
            # https://sushscience.wordpress.com/2016/12/04/understanding-alexnet/
            # https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637
            self.feature_extractor = AlexNetConv4()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.hidden_fc_layer = nn.Linear(in_features=256*13*13, out_features=300)
        else:
            self.hidden_fc_layer = nn.Linear(in_features=3*28*28, out_features=300)

        self.dropout = nn.Dropout(0.4)
        self.hidden_fc_layer_2 = nn.Linear(in_features=300, out_features=128)
        #self.hidden_fc_layer_3 = nn.Linear(in_features=512, out_features=128)
        self.flatten = Flatten()  # todo belki flatten kaldirilabilir
        self.final_layer = nn.Linear(in_features=128, out_features=class_number)
        # todo log softmax layer getirilebilir

    def get_learning_params(self):
        params_to_learn = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                # print (name, param.data) #debuging
                params_to_learn.append(param)

        return params_to_learn

    def forward(self, inp):
        if self.use_alexnet_extractor:
            inp = self.feature_extractor(inp)

        x = self.flatten(inp)
        x = torch.nn.functional.relu(self.hidden_fc_layer(x))
        if not self.use_alexnet_extractor:
            x = self.dropout(x)
        x = torch.nn.functional.relu(self.hidden_fc_layer_2(x))
        # x = torch.nn.functional.relu(self.hidden_fc_layer_3(x))
        x = self.final_layer(x)
        return x


class FaceRecognizerVisualizeModel(nn.Module):
    def __init__(self, class_number, use_cuda=True, use_alexnet_extractor=False):
        super(FaceRecognizerVisualizeModel, self).__init__()
        self.use_alexnet_extractor = use_alexnet_extractor
        self.use_cuda = use_cuda
        self.feature_extractor = None

        if use_alexnet_extractor:
            # https://sushscience.wordpress.com/2016/12/04/understanding-alexnet/
            # https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637
            self.feature_extractor = AlexNetConv4()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.hidden_fc_layer = nn.Linear(in_features=256*13*13, out_features=300)
        else:
            self.hidden_fc_layer = nn.Linear(in_features=3*28*28, out_features=300)
        self.flatten = Flatten()
        self.hidden_fc_layer_2 = nn.Linear(in_features=300, out_features=800)
        self.final_layer = nn.Linear(in_features=800, out_features=class_number)

    def get_learning_params(self):
        params_to_learn = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                # print (name, param.data) #debuging
                params_to_learn.append(param)

        return params_to_learn

    def forward(self, inp):
        if self.use_alexnet_extractor:
            inp = self.feature_extractor(inp)

        inp = self.flatten(inp)
        feature1 = torch.nn.functional.relu(self.hidden_fc_layer(inp))
        feature2 = torch.nn.functional.relu(self.hidden_fc_layer_2(feature1))
        x = self.final_layer(feature2)
        return x


class FaceRecognizerAlexNetModel(nn.Module):

    def __init__(self, class_number, use_cuda=True):
        super(FaceRecognizerAlexNetModel, self).__init__()

        self.use_cuda = use_cuda

        # todo testlerde son layer yerine sin bir kac layerda kullanilabilir
        self.alexnet = models.alexnet(pretrained=True)
        for param in self.alexnet.parameters():
            param.requires_grad = False
        self.alexnet.classifier[6] = nn.Linear(4096, class_number)

    def get_learning_params(self):
        params_to_learn = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                # print (name, param.data) #debuging
                params_to_learn.append(param)

        return params_to_learn

    def forward(self, inp):
        x = self.alexnet(inp)
        return x