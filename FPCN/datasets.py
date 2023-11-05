import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import  Image
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224,224), 'white')
    return img

class TestDataset(Dataset):
    def __init__(self, transform=None, dataloader=default_loader):
        self.data = []
        self.labels = []
        types = ['dolomite', 'marble', 'basalt']
        type_id = 0
        for type in types:
            type_imgs_path = os.path.join('rocks images samples', type, '*.jpg')
            type_imgs_path_list = glob.glob(type_imgs_path)
            for ip in type_imgs_path_list:
                self.data.append(ip)
            self.labels.extend([type_id] * len(type_imgs_path_list))
            type_id += 1
        self.tr_imgs, self.te_imgs, self.tr_labs, self.te_labs = train_test_split(self.data, self.labels, train_size=0.9, random_state=2)

        self.transform = transform
        self.dataloader = dataloader

        self.imglist = self.te_imgs

    def __getitem__(self, index):
        image_name = self.imglist[index]
        label = self.te_labs[index]
        image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label]


    def __len__(self):
        return len(self.imglist)

class TrainDataset(Dataset):
    def __init__(self, transform=None, dataloader=default_loader):
        self.data = []
        self.labels = []
        types = ['dolomite', 'marble', 'basalt']
        type_id = 0
        for type in types:
            type_imgs_path = os.path.join('rocks images samples', type, '*.jpg')
            type_imgs_path_list = glob.glob(type_imgs_path)
            for ip in type_imgs_path_list:
                self.data.append(ip)
            self.labels.extend([type_id] * len(type_imgs_path_list))
            type_id += 1
        self.tr_imgs, self.te_imgs, self.tr_labs, self.te_labs = train_test_split(self.data, self.labels, train_size=0.9, random_state=2)


        self.transform = transform
        self.dataloader = dataloader

        self.imglist = self.tr_imgs

        self.labels = self.tr_labs
        self.labels = np.array(self.labels)
        self.labels = torch.LongTensor(self.labels)


    def __getitem__(self, index):
        image_name = self.imglist[index]
        label = self.tr_labs[index]
        image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        label = torch.LongTensor([label])

        return [img, label]


    def __len__(self):
        return len(self.imglist)

class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
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
        return len(self.dataset) // self.batch_size