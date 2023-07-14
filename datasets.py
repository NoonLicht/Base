import torch
# import torch_directml
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from lib import transform


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

        # Поскольку каждое изображение может содержать разное количество объектов, нам нужна функция сопоставления (для передачи загрузчику данных).
        # Здесь описано, как объединить эти тензоры разных размеров. Мы используем списки.
        # Примечание: это не обязательно должно быть определено в этом классе, может быть автономным.
        # :param batch: повторяемый набор из N наборов из __getitem__()
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