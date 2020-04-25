
from torch import nn

import time
import torch
import tqdm


class ModelOperator:
    # todo accuracy ve epoch historyisine tutacak bi liste konulacak daha sonra analiz icin plot ettirelecek
    def __init__(self, model, criterion, tr_loader, val_loader, optimizer, num_epoch, patience, batch_size,
                 lr_scheduler=None):
        self.model = model
        self.tr_loader = tr_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.patience = patience
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.softmax = nn.LogSoftmax(dim=1)     # nn.Softmax() doest not overfit
        self.no_inc = 0
        self.best_loss = 9999
        self.phases = ["train", "val"]
        self.best_model = []
        self.best_val_acc = 0
        self.best_train_acc = 0
        self.best_val_loss = 0
        self.best_train_loss = 0
        self.batch_size = batch_size
        if self.model.use_cuda:
            self.model = self.model.cuda()


        self.train_accuracies = []
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []

    def train(self):

        pbar = tqdm.tqdm(desc= "Epoch 0, phase: Train", postfix="train_loss : ?, train_acc: ?")
        start = time.time()
        for i in range(self.num_epoch):
            last_train_acc = 0
            last_val_acc = 0
            last_val_loss = 0
            last_train_loss = 0
            pbar.update(1)
            for phase in self.phases:
                total_acc = 0
                total_loss = 0

                if phase == "train":
                    pbar.set_description_str("Epoch %d,"% i + "phase: Training")
                    loader = self.tr_loader
                    self.model.train()
                else:
                    pbar.set_description_str("Epoch %d,"% i + "phase: Validation")
                    loader = self.val_loader
                    self.model.eval()
                iter = 0

                for images, labels in loader:
                    iter += 1
                    if self.model.use_cuda:
                        #images, labels = (Variable(images)).cuda(), (Variable(labels)).cuda()
                        images = images.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)
                    self.optimizer.zero_grad()
                    logits = self.model(images)
                    softmaxed_scores = self.softmax(logits)
                    _, predictions = torch.max(softmaxed_scores, 1)
                    if labels.ndim > 1:
                        _, labels = torch.max(labels, 1)
                    loss = self.criterion(softmaxed_scores, labels)

                    curr_loss = loss.item()
                    total_loss += curr_loss

                    curr_acc = torch.sum(predictions == labels).item()
                    total_acc += curr_acc

                    if phase == "train":
                        pbar.set_postfix_str("train acc: %6.3f," %(total_acc/ (iter*self.batch_size)) + ("train loss: %6.3f" % (total_loss / iter)))
                        loss.backward()
                        self.optimizer.step()
                    else:
                        pbar.set_postfix_str("val acc: %6.3f," %(total_acc/ (iter*self.batch_size)) + ("val loss: %6.3f" % (total_loss / iter)))

                if phase == "train":
                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                if phase == "train":
                    loss_p = total_loss / iter
                    acc_p = total_acc / len(self.tr_loader.dataset)
                    last_train_acc = acc_p
                    last_train_loss = loss_p

                    self.train_accuracies.append(last_train_acc)
                    self.train_losses.append(last_train_loss)
                else:
                    loss_p = total_loss / iter
                    acc_p = total_acc / len(self.val_loader.dataset)
                    last_val_acc = acc_p
                    last_val_loss = loss_p

                    self.val_accuracies.append(last_val_acc)
                    self.val_losses.append(last_val_loss)

                    if loss_p < self.best_loss:
                        print(phase, "\t:New best loss, loss is: ",str(loss_p), "acc is: ",acc_p )
                        self.best_loss = loss_p
                        self.no_inc = 0
                        self.best_model = self.model
                        self.best_train_acc = last_train_acc
                        self.best_train_loss = last_train_loss
                        self.best_val_loss = last_val_loss
                        self.best_val_acc = last_val_acc
                    else:
                        print("Not a better score")

                        self.no_inc += 1
                        if self.no_inc == self.patience:
                            print("Out of patience returning the best model")
                            print(
                                "Best val acc: {}, Best val loss: {}, Best train acc: {}, Best train loss: {} ".format(
                                    self.best_val_acc, self.best_val_loss, self.best_train_acc, self.best_train_loss
                                ))  # Stats of the best model
                            return self.best_model
        end = time.time()
        print("Elapsed Time: " + str(end - start))
        print("Training ended returning the best model")
        print(
            "Best val acc: {}, Best val loss: {}, Best train acc: {}, Best train loss: {} ".format(
                self.best_val_acc, self.best_val_loss, self.best_train_acc, self.best_train_loss
            ))  # Stats of the best model
        return self.best_model

    def test(self, test_loader):

        with torch.no_grad():
            total_acc = 0
            self.model.eval()
            for images, labels in test_loader:
                if self.model.use_cuda:
                    # images, labels = (Variable(images)).cuda(), (Variable(labels)).cuda()
                    images = images.cuda()
                    labels = labels.cuda()
                logits = self.model(images)
                softmaxed_scores = self.softmax(logits)
                _, predictions = torch.max(softmaxed_scores, 1)

                if labels.ndim > 1:
                    _, labels = torch.max(labels, 1)

                #loss = self.criterion(softmaxed_scores, labels)
                #test_loss += loss.item()
                total_acc += torch.sum(predictions == labels).item()

            print('Accuracy of the network on the test images: %d %%' % (
                100 * total_acc / len(test_loader.dataset)))

    def get_training_history(self):
        return {"training_acc": self.train_accuracies,
                "training_loss": self.train_losses,
                "val_acc": self.val_accuracies,
                "val_loss": self.val_losses}