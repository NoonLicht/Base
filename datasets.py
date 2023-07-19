import torch
from torch.utils.data import Dataset
import json
import os
import h5py
from PIL import Image
from lib import transform
from lib_ic import *


class PascalVOCDataset(Dataset):
    
    # Класс набора данных высоты тона, который будет использоваться в загрузчике данных высоты тона для создания пакетов.

    def __init__(self, data_folder, split, keep_difficult=False):
        
        # :параметр data_folder: папка, в которой хранятся файлы данных
        # :параметр split: разделение, одно из "ОБУЧАТЬ" или "ТЕСТИРОВАТЬ"
        # :параметр keep_difficult: сохранять или отбрасывать объекты, которые считаются трудноопределимыми?
        
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Чтение файлов
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Чтение фотографий
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Считывание объектов на этом изображении (ограничивающие рамки, надписи, помехи)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Выброс сложных предметов
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Преобразование фотографий, боксов, надписей и помех
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):

        # :параметр batch: повторяемый набор из N наборов из __getitem__()
        # :return: тензор изображений, списки тензоров ограничивающих рамок различного размера, меток и трудностей

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # тензор (N, 3, 300, 300)
    

class CaptionDataset(Dataset):

    # Класс набора данных Pitch, который будет использоваться в загрузчике данных PyTorch для создания пакетов.

    def __init__(self, data_folder, data_name, split, transform=None):

        # :параметр data_folder: папка, в которой хранятся файлы данных
        # :параметр data_name: базовое имя обработанных наборов данных
        # :параметр split: разделить, один из 'TRAIN', 'VAL' или 'TEST'
        # :параметр transform: конвейер преобразования изображений

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Откроет файл hdf5, в котором хранятся изображения
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Подписи к каждому изображению
        self.cpi = self.h.attrs['captions_per_image']

        # Загружает закодированные подписи (полностью в память)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Загружает длины подписей (полностью в память)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Конвейер преобразования PyTorch для изображения (нормализация и т.д.)
        self.transform = transform

        # Общее количество точек данных
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # N-й заголовок соответствует (N // captions_per_image)-му изображению
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # Для проверки результатов тестирования также возвращает все подписи 'captions_per_image', чтобы найти оценку BLEU-4
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
