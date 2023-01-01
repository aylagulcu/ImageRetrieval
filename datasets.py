import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import cv2
import numpy as np
import os

class DeepFashionDataset(Dataset):
    def __init__(self,img_dir,train_path=None,test_path=None,validation_path=None,mode="train", transform=None, ):
        
        self.transform= transform
        self.img_dir = img_dir

        if mode=="train":
            assert(train_path is not None)
            self.file_list = train_path
        elif mode=="test":
            assert(test_path is not None)
            self.file_list = test_path
        elif mode=="validation":
            assert(validation_path is not None)
            self.file_list = validation_path
        else:
            return
        
        df = pd.read_csv(self.file_list, header=0)
        
        self.data = df["file"].to_numpy().tolist()
        self.main_classes = df["main_category"].to_numpy().tolist() # class id of each sample
        self.sub_classes = df["sub_category"].to_numpy().tolist()
        self.clothes_types = df["clothes_type"].to_numpy().tolist()
        self.source_types = df["source_type"].to_numpy().tolist()
        self.variation_types=df["variation_type"].to_numpy().tolist()
        self.bboxes = df["bbox"].to_numpy().tolist()        
        self.landmarks = df["landmarks"].to_numpy().tolist()
        self.attributes = df["attributes"].to_numpy().tolist()
        self.classes = [i+"/"+j for i,j in zip(self.main_classes,self.sub_classes)  ]
        self.classes_set = set(self.classes)
        
        del df

        #
        # ["file","main_category","sub_category","clothes_type","source_type","variation_type","bbox","landmarks","attributes"]
        # with open(self.file_list,"r") as file : 
        #     lines = file.readlines()
        #     for line in lines[1:]:
        #         splitted_data = line.split(",")
        #         self.data.append(splitted_data[0])
        #         self.main_labels.append(splitted_data[1])
        #         self.sub_labels.append(splitted_data[2])
        #         self.clothes_types.append(splitted_data[3])
        #         self.source_types.append(splitted_data[4])
        #         self.variation_types.append(splitted_data[5])
        #         self.bboxes.append(eval(splitted_data[6]))
        #         self.landmarks.append(eval(splitted_data[7]))
        #         self.attributes.append(eval(splitted_data[8]))


        # for class_path in self.file_list:
        #     class_name = class_path.split("/")[-1]
        #     self.classes.append(class_name)
        #     for img_path in glob.glob(class_path + "/*.jpg"):
        #         self.data.append([img_path, class_name])
                              
        self.idx_to_class = {i:j for i, j in enumerate(self.classes)} # given id, get class name
        
        self.class_to_label= {cls:lab for lab,cls in enumerate(self.classes_set)} # given class name, get label (int)
        self.labels= [self.class_to_label[i] for i in self.classes]
        self.labels= np.array(self.labels)
        self.labels_set = set(self.labels)
        
        self.class_to_indices = {cls: np.squeeze(np.where(np.array(self.classes) == cls)).tolist() 
                                 for cls in self.classes_set} # given class name, get list of item indices
        self.label_to_indices = {label: np.squeeze(np.where(self.labels == label))
                                 for label in self.labels_set} # given class name, get list of item indices  
        
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(os.path.join(self.img_dir,*img_path.split("/")))
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        
        c, w, h = img_tensor.shape
        if w< 300:
            transform = transforms.Pad((0, 0, 0, 300-w)) # top, left, bottom, right
            img_tensor = transform(img_tensor)
        elif h<300:
            transform = transforms.Pad((0, 0, 300-h, 0)) 
            img_tensor = transform(img_tensor)
            

        if self.transform is not None:
            image = self.transform(img_tensor)
        else :
            image = img_tensor
        return image, self.class_to_label[self.classes[idx]]

    

class TripletDeepFashion(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, inner_dataset, mode="train"):
        self.inner_dataset = inner_dataset
        self.mode= mode
        
        self.data = self.inner_dataset.data
        self.classes= self.inner_dataset.classes
        self.labels= self.inner_dataset.labels
        self.labels_set= self.inner_dataset.labels_set
        
        self.class_to_indices= self.inner_dataset.class_to_indices # given class name, get list of item indices
        self.label_to_indices = self.inner_dataset.label_to_indices # given label, get list of item indices
        
        
        if mode != "train":
            # generate fixed triplets for testing

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.data))]
            self.triplets = triplets

    def __getitem__(self, index):
        if self.mode == "train":
            #img1, label1 = self.data[index], self.labels[index].item()
            img1, label1 = self.inner_dataset.__getitem__(index)
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            img2, label2= self.inner_dataset.__getitem__(positive_index)
            img3, label3= self.inner_dataset.__getitem__(negative_index)

            return (img1, img2, img3), [label1, label2, label3]
        else:
            img1, label1= self.inner_dataset.__getitem__(self.triplets[index][0])
            img2, label2 = self.inner_dataset.__getitem__(self.triplets[index][1])
            img3, label3= self.inner_dataset.__getitem__(self.triplets[index][2])

            return (img1, img2, img3), [label1, label2, label3]

    def __len__(self):
        return len(self.inner_dataset)











import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import torch
from torch import nn

import glob
import os 
import csv
import cv2


#######################################################
#               Define Dataset Class
#######################################################
class CarsDataset(Dataset):
    def __init__(self, transform=None, train=True):
        self.transform= transform
        
        if train==True:
            self.imgs_path = "/home/ayla/data/Stanford_Cars_Dataset/train/"
        else:
            self.imgs_path = "/home/ayla/data/Stanford_Cars_Dataset/test/"
        
        self.file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        self.classes= [] # class names
        self.labels= [] # class id of each sample
        
        for class_path in self.file_list:
            class_name = class_path.split("/")[-1]
            self.classes.append(class_name)
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
                
        self.idx_to_class = {i:j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}
        self.labels= [self.class_to_idx[j] for i,j in self.data]
        self.labels= np.array(self.labels)

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        class_id = self.class_to_idx[class_name]
        img = cv2.imread(img_path)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        
        if self.transform is not None:
            image = self.transform(img_tensor)

        return image, class_id

        
        

#######################################################
#               Define Dataset Class
#######################################################

class CarsDatasetSamed(Dataset):
    def __init__(self,root, transform=None):
        self.transform= transform
        self.imgs_path = os.path.join(root,"car_data","car_data","train")
        self.anno_path = os.path.join(root,"anno_train.csv")

        reader = csv.reader(open(self.anno_path, newline=''), delimiter=',', quotechar='"')
        annotations = {}
        for row in reader : 
            annotations[row[0]] = {"x":row[1],"y":row[2],"w":row[3],"h":row[4],"class_name":row[5]}

        
        self.data = []
        self.classes_to_id = []
        for class_path in os.listdir(self.imgs_path):
            print(class_path)
            for image_files in os.listdir(os.path.join(self.imgs_path,class_path)):
                self.data.append({"image": cv2.imread(os.path.join(self.imgs_path,class_path,image_files)) ,"x": annotations[image_files]["x"],"y": annotations[image_files]["y"],"w": annotations[image_files]["w"],"h": annotations[image_files]["h"],"class_name": annotations[image_files]["class_name"] })
                
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        selected_data = self.data[idx]

        img_tensor = torch.from_numpy(selected_data["image"])
        img_tensor = img_tensor.permute(2, 0, 1)
        
        if self.transform is not None:
            image = self.transform(img_tensor)

        return image, selected_data["class_name"]



    
class TripletCars(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, inner_dataset, train=True):
        self.inner_dataset = inner_dataset
        self.train= train
        
        self.classes = self.inner_dataset.classes
        self.data = self.inner_dataset.data
        self.labels= self.inner_dataset.labels
        self.labels_set = set(self.labels)
        # keeps the sample indices belonging to each label:
        self.label_to_indices = {label: np.squeeze(np.where(self.labels == label))
                                 for label in self.labels_set}

        if not self.train:
            # generate fixed triplets for testing

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.data))]
            self.triplets = triplets

    def __getitem__(self, index):
        if self.train:
            #img1, label1 = self.data[index], self.labels[index].item()
            img1, label1 = self.inner_dataset.__getitem__(index)
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            img2, label2= self.inner_dataset.__getitem__(positive_index)
            img3, label3= self.inner_dataset.__getitem__(negative_index)

            return (img1, img2, img3), [label1, label2, label3]
        else:
            img1, label1= self.inner_dataset.__getitem__(self.triplets[index][0])
            img2, label2 = self.inner_dataset.__getitem__(self.triplets[index][1])
            img3, label3= self.inner_dataset.__getitem__(self.triplets[index][2])

            return (img1, img2, img3), [label1, label2, label3]

    def __len__(self):
        return len(self.inner_dataset)
    

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        
       

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
