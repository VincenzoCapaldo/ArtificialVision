import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from .reid_dataset import import_MarketDuke_nodistractors
from .reid_dataset import import_Market1501Attribute_binary
from .reid_dataset import import_DukeMTMCAttribute_binary


class Train_Dataset(data.Dataset):

    def __init__(self, data_dir, dataset_name, transforms=None, train_val='train'):

        train, query, gallery = import_MarketDuke_nodistractors(data_dir, dataset_name)

        if dataset_name == 'Market-1501':
            train_attr, test_attr, self.label = import_Market1501Attribute_binary(data_dir)
        elif dataset_name == 'DukeMTMC-reID':
            train_attr, test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        else:
            print('Input should only be Market1501 or DukeMTMC')

        self.num_ids = len(train['ids'])
        self.num_labels = len(self.label)

        # distribution:每个属性的正样本占比
        distribution = np.zeros(self.num_labels)
        for k, v in train_attr.items():
            distribution += np.array(v)
        self.distribution = distribution / len(train_attr)

        if train_val == 'train':
            self.train_data = train['data']
            self.train_ids = train['ids']
            self.train_attr = train_attr
        elif train_val == 'query':
            self.train_data = query['data']
            self.train_ids = query['ids']
            self.train_attr = test_attr
        elif train_val == 'gallery':
            self.train_data = gallery['data']
            self.train_ids = gallery['ids']
            self.train_attr = test_attr
        else:
            print('Input should only be train or val')

        self.num_ids = len(self.train_ids)

        if transforms is None:
            if train_val == 'train':
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.train_data[index][0]
        i = self.train_data[index][1]
        id = self.train_data[index][2]
        cam = self.train_data[index][3]
        label = np.asarray(self.train_attr[id])
        data = Image.open(img_path)
        data = self.transforms(data)
        name = self.train_data[index][4]
        return data, i, label, id, cam, name

    def __len__(self):
        return len(self.train_data)

    def num_label(self):
        return self.num_labels

    def num_id(self):
        return self.num_ids

    def labels(self):
        return self.label


class Test_Dataset(data.Dataset):
    def __init__(self, data_dir, dataset_name, transforms=None, query_gallery='query'):
        train, query, gallery = import_MarketDuke_nodistractors(data_dir, dataset_name)

        if dataset_name == 'Market-1501':
            self.train_attr, self.test_attr, self.label = import_Market1501Attribute_binary(data_dir)
        elif dataset_name == 'DukeMTMC-reID':
            self.train_attr, self.test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        else:
            print('Input should only be Market1501 or DukeMTMC or Custom-dataset')

        if query_gallery == 'query':
            self.test_data = query['data']
            self.test_ids = query['ids']
        elif query_gallery == 'gallery':
            self.test_data = gallery['data']
            self.test_ids = gallery['ids']
        elif query_gallery == 'all':
            self.test_data = gallery['data'] + query['data']
            self.test_ids = gallery['ids']
        else:
            print('Input shoud only be query or gallery;')

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(size=(288, 144)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.test_data[index][0]
        id = self.test_data[index][2]
        label = np.asarray(self.test_attr[id])
        data = Image.open(img_path)
        data = self.transforms(data)
        name = self.test_data[index][4]
        return data, label, id, name

    def __len__(self):
        return len(self.test_data)

    def labels(self):
        return self.label


class Train_Custom_Dataset(data.Dataset):
    def __init__(self, data_dir, val_ratio=0.2, transforms=None, train_val='train'):
        """
        Classe per il custom dataset (train/validation split).
        :param data_dir: percorso della directory del dataset
        :param val_ratio: proporzione di dati riservata per la validazione
        :param transforms: trasformazioni da applicare alle immagini
        :param train_val: specifica se caricare il train o la validation (val)
        """
        training_set_path = os.path.join(data_dir, 'training_set.txt')

        # Legge i dati e divide in train/val
        train_data = []
        val_data = []
        train_attr = {}
        val_attr = {}

        with open(training_set_path, 'r') as f:
            lines = f.readlines()
            np.random.shuffle(lines)  # Shuffle per mescolare i dati
            split_idx = int(len(lines) * (1 - val_ratio))

            for i, line in enumerate(lines):
                parts = line.strip().split(',')
                img_name, labels = parts[0], list(map(int, parts[3:]))
                img_path = os.path.join(data_dir, 'training_set', img_name)

                if i < split_idx:  # Train
                    train_data.append([img_path, img_name])  # Salva percorso immagine e nome file
                    train_attr[img_name] = labels
                else:  # Validation
                    val_data.append([img_path, img_name])  # Salva percorso immagine e nome file
                    val_attr[img_name] = labels

        self.data = train_data if train_val == 'train' else val_data
        self.attr = train_attr if train_val == 'train' else val_attr
        self.num_labels = 3  # Gender, Bag, Hat

        # Trasformazioni
        if transforms is None:
            if train_val == 'train':
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        """
        Restituisce i dati di una singola immagine.
        :param index: indice dell'immagine
        :return: data, index, label, id (None), cam (None), name
        """
        img_path, name = self.data[index]  # Ottieni percorso immagine e nome
        label = np.asarray(self.attr[name])  # Etichette binarie
        data = Image.open(img_path)  # Carica immagine
        data = self.transforms(data)  # Applica trasformazioni

        # Ritorna id e cam come None, e il nome del file come name
        id = ""
        cam = ""

        return data, index, label, id, cam, name

    def __len__(self):
        return len(self.data)

    def labels(self):
        return ['Gender', 'Bag', 'Hat']

    def num_label(self):
        return self.num_labels


class Test_Custom_Dataset(data.Dataset):
    def __init__(self, data_dir, transforms=None):
        """
        Classe per il custom dataset (test set unificato).
        :param data_dir: percorso della directory del dataset
        :param transforms: trasformazioni da applicare alle immagini
        """
        test_set_path = os.path.join(data_dir, 'test_set.txt')

        # Legge i dati
        test_data = []
        test_attr = {}

        with open(test_set_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                img_name, labels = parts[0], list(map(int, parts[2:]))
                img_path = os.path.join(data_dir, 'test_set', img_name)
                test_data.append([img_path, img_name])  # Salva percorso immagine e nome file
                test_attr[img_name] = labels


        self.data = test_data
        self.attr = test_attr
        self.num_labels = 3  # Gender, Bag, Hat

        # Trasformazioni
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(size=(288, 144)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        """
        Restituisce i dati di una singola immagine.
        :param index: indice dell'immagine
        :return: data, label, id (None), name
        """
        img_path, name = self.data[index]  # Ottieni percorso immagine e nome
        label = np.asarray(self.attr[name])  # Etichette binarie
        data = Image.open(img_path)  # Carica immagine
        data = self.transforms(data)  # Applica trasformazioni

        # Ritorna id come None, e il nome del file come name
        id = "None"

        return data, label, id, name

    def __len__(self):
        return len(self.data)

    def labels(self):
        return ['Gender', 'Bag', 'Hat']

    def num_label(self):
        return self.num_labels
