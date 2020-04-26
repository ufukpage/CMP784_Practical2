from pylab import *
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
"""
import cv2
import os
from os import listdir
from os.path import isfile, join
import shutil
import urllib
from sklearn.model_selection import train_test_split


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


def downloadSubset(file_name):
    act = list(set([a.split("\t")[0] for a in open(file_name).readlines()]))
    testfile = urllib.request.URLopener()
    bbox_dict = {}
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(file_name):
            if a in line:
                filename = name + str(i) + '.' + line.split()[4].split('.')[-1]

                bboxlist = line.split()[5].split(',')
                # A version without timeout (uncomment in case you need to
                # unsupress exceptions, which timeout() does)
                # testfile.retrieve(line.split()[4], "uncropped/"+filename)
                # timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/" + filename), {}, 30)
                if not os.path.isfile("uncropped/" + filename):
                    continue
                print(filename)
                bbox_dict[filename] = bboxlist
                i += 1
    return bbox_dict


def cropImage(bbox_dict):

    damaged_image = []

    for file_name in bbox_dict:
        bboxlist = bbox_dict[file_name]
        bboxlist = [int(i) for i in bboxlist]
        image = cv2.imread("uncropped/" + file_name)
        if image is None or image.size == 0:
            damaged_image.append(file_name)
            continue
        image = image[bboxlist[1]:bboxlist[3], bboxlist[0]:bboxlist[2]]
        if image.size == 0:
            damaged_image.append(file_name)
            continue
        # cv2.imshow(file_name, image)
        # cv2.waitKey(0)
        cv2.imwrite("cropped/" + file_name, image)
    return damaged_image


def cropImage(bbox_dict):

    damaged_image = []

    for file_name in bbox_dict:
        bboxlist = bbox_dict[file_name]
        bboxlist = [int(i) for i in bboxlist]
        image = cv2.imread("uncropped/" + file_name)
        if image is None or image.size == 0:
            damaged_image.append(file_name)
            continue
        image = image[bboxlist[1]:bboxlist[3], bboxlist[0]:bboxlist[2]]
        if image.size == 0:
            damaged_image.append(file_name)
            continue
        # cv2.imshow(file_name, image)
        # cv2.waitKey(0)
        cv2.imwrite("cropped/" + file_name, image)
    return damaged_image


def split_class_dataset_randomly(class_set, test_ratio=0.1, val_ratio=0.1):
    data_size = len(class_set)
    x_train, x_test, _, _ = train_test_split(class_set, np.full(data_size, 0), test_size=test_ratio, random_state=42)

    val_length = val_ratio * len(x_test) / test_ratio
    val_ratio = val_length / len(x_train)
    if val_ratio > 1:
        raise Exception('Ratio invalid')
    x_train, x_val, _, _ = train_test_split(x_train,  np.full(len(x_train), 0), test_size=val_ratio, random_state=None)
    return x_train, x_test, x_val


def create_data_set(cropped_image_path):
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    root_dataset = os.path.join(current_path, "dataset_part2")
    train_set_path = os.path.join(root_dataset, "train")
    test_set_path = os.path.join(root_dataset, "test")
    val_set_path = os.path.join(root_dataset, "val")

    test_ratio = 0.1
    val_ratio = 0.1

    if not os.path.exists(root_dataset):
        os.mkdir(root_dataset)
    if not os.path.exists(train_set_path):
        os.mkdir(train_set_path)
    if not os.path.exists(test_set_path):
        os.mkdir(test_set_path)
    if not os.path.exists(val_set_path):
        os.mkdir(val_set_path)

    data_set = {}
    for f in listdir(cropped_image_path):
        if isfile(join(cropped_image_path, f)):
            """"
            file_name = ''.join([i for i in f if not i.isdigit()])
            if file_name in data_set_counter:
                data_set_counter[file_name] += 1
            else:
                data_set_counter[file_name] += 0

            if data_set_counter[file_name] < 70:
                if file_name in train_image:
                    train_image[file_name].append(join(cropped_image_path, f))
                else:
                    train_image = []
            """
            ############################
            class_name = (''.join([i for i in f if not i.isdigit()])).split(".")[0]
            if class_name in data_set:
                data_set[class_name].append(join(cropped_image_path, f))
            else:
                data_set[class_name] = [join(cropped_image_path, f)]

    for label in data_set:
        print(len(data_set[label]))

    for label in data_set:
        train_set, test_set, val_set = split_class_dataset_randomly(data_set[label],
                                                                    test_ratio=test_ratio, val_ratio=val_ratio)
        train_class_dir = os.path.join(train_set_path, str(label))
        test_class_dir = os.path.join(test_set_path, str(label))
        val_class_dir = os.path.join(val_set_path, str(label))
        if not os.path.exists(train_class_dir):
            os.mkdir(train_class_dir)
        if not os.path.exists(test_class_dir):
            os.mkdir(test_class_dir)
        if not os.path.exists(val_class_dir):
            os.mkdir(val_class_dir)

        for train_sample_path in train_set:
            shutil.copy(train_sample_path, train_class_dir)

        for test_sample_path in test_set:
            shutil.copy(test_sample_path, test_class_dir)

        for val_sample_path in val_set:
            shutil.copy(val_sample_path, val_class_dir)



def create_data_set_new(cropped_image_path):
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    root_dataset = os.path.join(current_path, "dataset_part2")
    train_set_path = os.path.join(root_dataset, "train")
    test_set_path = os.path.join(root_dataset, "test")
    val_set_path = os.path.join(root_dataset, "val")

    test_ratio = 0.1
    val_ratio = 0.1

    if not os.path.exists(root_dataset):
        os.mkdir(root_dataset)
    if not os.path.exists(train_set_path):
        os.mkdir(train_set_path)
    if not os.path.exists(test_set_path):
        os.mkdir(test_set_path)
    if not os.path.exists(val_set_path):
        os.mkdir(val_set_path)

    data_set = {}
    celebs = listdir(cropped_image_path)
    for class_name in celebs:
        class_path = os.path.join(cropped_image_path, class_name)

        data_set[class_name] = []

        for root, dirs, files in os.walk(class_path):

            for file in files:
                # print(os.path.join(class_path, file))
                data_set[class_name].append(os.path.join(class_path, file))

    for label in data_set:
        print(len(data_set[label]))

    for label in data_set:
        train_set, test_set, val_set = split_class_dataset_randomly(data_set[label],
                                                                    test_ratio=test_ratio, val_ratio=val_ratio)
        train_class_dir = os.path.join(train_set_path, str(label))
        test_class_dir = os.path.join(test_set_path, str(label))
        val_class_dir = os.path.join(val_set_path, str(label))
        if not os.path.exists(train_class_dir):
            os.mkdir(train_class_dir)
        if not os.path.exists(test_class_dir):
            os.mkdir(test_class_dir)
        if not os.path.exists(val_class_dir):
            os.mkdir(val_class_dir)

        for train_sample_path in train_set:
            shutil.copy(train_sample_path, train_class_dir)

        for test_sample_path in test_set:
            shutil.copy(test_sample_path, test_class_dir)

        for val_sample_path in val_set:
            shutil.copy(val_sample_path, val_class_dir)

if __name__ == "__main__":
    """
    if not os.path.exists("uncropped"):
        os.mkdir("uncropped")
    if not os.path.exists("cropped"):
        os.mkdir("cropped")
    #####################################
    
    file_name = 'male_subset_actors.txt'
    bbox_male = downloadSubset(file_name)
    np.save("bbox_male.npy", [bbox_male])

    #bbox_male = np.load("bbox_male.npy", allow_pickle=True)
    damaged_image = cropImage(bbox_male)
    np.save("damaged_image_male.npy", [damaged_image])
   
    #####################################

    file_name = 'female_subset_actors.txt'
    bbox_female = downloadSubset(file_name)
    np.save("bbox_female.npy", bbox_female)
    damaged_image = cropImage(bbox_female)
    np.save("damaged_image_female.npy", [damaged_image])
    """
    #cropped_image_path = "faces"
    #create_data_set(cropped_image_path)
    create_data_set_new("faces")
