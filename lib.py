import json
import os
import torch
# import torch_directml
import random
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from collections import Counter
from random import *
import h5py
import cv2
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch_directml.device()

# Карта надписей
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):

    # Создаем списки изображений, ограничивающих рамок и меток объектов на этих изображениях и сохраните их в файл.

    # ::параметр voc07_path: путь к папке 'VOC 2007'
    # ::параметр vol 12_path: путь к папке 'VOC 2012'
    # :параметр output_folder: папка, в которой должны быть сохранены JSON-файлы  

    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Обучающие данные
    for path in [voc07_path, voc12_path]:

        # Поиск идентификаторов изображений в обучающих данных
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Парсинг XML-файл аннотаций
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Сохранение
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Тест
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Найдите идентификаторы изображений в тестовых данных
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Парсинг XML-файл аннотаций
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Сохранение
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


def decimate(tensor, m):

    # Уменьшим тензор на коэффициент 'm', т.е. уменьшим выборку, сохранив каждое 'm-е значение.

    # Это используется, когда мы преобразуем слои FC в эквивалентные сверточные слои, но меньшего размера.

    # :параметр tensor: тензор, подлежащий уничтожению
    # :параметр m: список коэффициентов прореживания для каждого измерения тензора; Нет, если не требуется прореживание по измерению
    # :return: уменьшенный тензор
    
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):

    # Вычисляем среднюю точность (mAP) обнаруженных объектов.

    # :параметр det_boxes: список тензоров, по одному тензору для каждого изображения, содержащего ограничивающие рамки обнаруженных объектов
    # :параметр det_labels: список тензоров, по одному тензору для каждого изображения, содержащего метки обнаруженных объектов
    # :параметр det_scores: список тензоров, по одному тензору для каждого изображения, содержащего оценки меток обнаруженных объектов
    # :параметр true_boxes: список тензоров, по одному тензору для каждого изображения, содержащего ограничивающие рамки реальных объектов
    # :параметр true_labels: список тензоров, по одному тензору для каждого изображения, содержащего метки реальных объектов
    # :параметр true_difficulties: список тензоров, по одному тензору для каждого изображения, содержащего сложность реальных объектов (0 или 1)
    # :return: список средних значений точности для всех классов, средняя точность (карта)

    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # все это списки тензоров одинаковой длины, т.е. количество изображений
    n_classes = len(label_map)

    # Храним все (истинные) объекты в одном непрерывном тензоре, отслеживая изображение, из которого они взяты
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects это общее количество объектов на всех изображениях
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Сохраняем все обнаруженные данные в одном непрерывном тензоре, отслеживая изображение, с которого они получены
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Рассчитываем точки доступа для каждого класса (кроме фонового)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Извлекаем только объекты с этим классом
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Следим за тем, какие истинные объекты с этим классом уже были "обнаружены"
        # Пока что ни одного
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Извлекать только обнаружения с помощью этого класса
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Сортировка обнаружений в порядке убывания достоверности/баллов
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # В порядке убывания баллов проверьте, является ли результат истинным или ложноположительным
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Найдем объекты на одном изображении с этим классом, их трудности и были ли они обнаружены ранее
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # Если такого объекта на этом изображении нет, то обнаружение является ложноположительным
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Найдем максимальное совпадение этого обнаружения с объектами на этом изображении этого класса
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' - это индекс объекта в этих тензорах уровня изображения 'object_boxes', 'object_difficulties'
            # В исходных тензорах уровня класса 'true_class_boxes' и т.д. 'ind' соответствует объекту с индексом...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # Нам нужен 'original_find' для обновления 'true_class_boxes_detectedd'

            # Если максимальное перекрытие превышает пороговое значение 0,5, это совпадение
            if max_overlap.item() > 0.5:
                # Если объект, с которым он сопоставлен, "сложный", игнорируем его
                if object_difficulties[ind] == 0:
                    # Если этот объект еще не был обнаружен, это истинно положительный результат
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # теперь этот объект обнаружен/учтен
                    # В противном случае это ложноположительный результат (поскольку этот объект уже учтен)
                    else:
                        false_positives[d] = 1
            # В противном случае обнаружение происходит в местоположении, отличном от фактического объекта, и является ложноположительным
            else:
                false_positives[d] = 1

        # Вычисляем совокупную точность и вспоминаем при каждом обнаружении в порядке уменьшения баллов
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Найдите среднее значение максимальной точности, соответствующей отзывам выше порогового значения "t".
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Вычислить среднюю точность (карта)
    mean_average_precision = average_precisions.mean().item()

    # Сохраняем в словаре среднюю точность по классам
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy(xy):

    # Преобразуем ограничивающие рамки из граничных координат (x_min, y_min, x_max, y_max_max) в координаты центрального размера (c_x, c_y, w, h).

    # :параметр xy: ограничивающие прямоугольники в граничных координатах, тензор размера (n_boxes, 4)
    # :return: ограничивающие прямоугольники в координатах центрального размера, тензор размера (n_boxes, 4)

    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):

    # Преобразуем ограничивающие рамки из координат центрального размера (c_x, c_y, w, h) в граничные координаты (x_min, y_min, x_max, y_max_max).

    # :параметр cx cy: ограничивающие прямоугольники в координатах центрального размера, тензор размера (n_boxes, 4)
    # :return: ограничивающие прямоугольники в координатах границ, тензор размера (n_boxes, 4)

    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):

    # Закодируем ограничивающие прямоугольники (которые находятся в форме центрального размера) с соответствующими предыдущими прямоугольниками (которые находятся в форме центрального размера).

    # Для координат центра найдем смещение относительно предыдущего прямоугольника и масштабируем по размеру предыдущего прямоугольника.
    # Для получения координат размера масштабируем по размеру предыдущего поля и преобразуем в логарифмическое пространство.

    # В модели мы предсказываем координаты ограничивающего прямоугольника в этой закодированной форме.

    # :параметр cx cy: ограничивающие рамки в координатах центрального размера, тензор размера (n_priors, 4)
    # :параметр priors_cx cy: предыдущие поля, относительно которых должно быть выполнено кодирование, тензор размера (n_priors, 4)
    # :return: закодированные ограничивающие рамки, тензор размера (n_priors, 4)

    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):

    # Расшифруем координаты ограничивающего прямоугольника, предсказанные моделью, поскольку они закодированы в форме, упомянутой выше.

    # Они декодируются в координаты центрального размера.

    # Это обратная функция приведенной выше.

    # :параметр gcxgcy: закодированные ограничивающие рамки, т.е. выходные данные модели, тензор размера (n_priors, 4)
    # :параметр priors_cx cy: предыдущие блоки, относительно которых определена кодировка, тензор размера (n_priors, 4)
    # :return: декодированные ограничивающие рамки в форме центрального размера, тензор размера (n_priors, 4)


    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):

    # Найдем пересечение каждой комбинации блоков между двумя наборами блоков, которые находятся в граничных координатах.

    # :параметр set_1: набор 1, тензор измерений (n1, 4)
    # :параметр set_2: набор 2, тензор измерений (n2, 4)
    # :return: пересечение каждого из прямоугольников в наборе 1 относительно каждого из прямоугольников в наборе 2, тензор измерений (n1, n2)

    # PyTorch автоматически транслирует одноэлементные измерения
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):

    # Найдем перекрытие Жаккарда (Вы) каждой комбинации блоков между двумя наборами блоков, которые находятся в граничных координатах.

    # :параметр set_1: набор 1, тензор измерений (n1, 4)
    # :параметр set_2: набор 2, тензор измерений (n2, 4)
    # :return: Перекрытие Жаккарда каждого из блоков в наборе 1 относительно каждого из блоков в наборе 2, тензор размеров (n1, n2)


    # Найдем пересечения
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Найдем области каждой ячейки в обоих наборах
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # PyTorch автоматически транслирует одноэлементные измерения
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def expand(image, boxes, filler):

    # Выполним операцию уменьшения масштаба, поместив изображение на холст большего размера из наполнителя.

    # Помогает научиться обнаруживать объекты меньшего размера.

    # :параметр image: изображение, тензор измерений (3, original_h, original_w)
    # :параметр box: ограничивающие рамки в граничных координатах, тензор измерений (n_objects, 4)
    # :параметр filler: значения RBG для материала наполнителя, список типа [R, G, B]
    # :return: расширенное изображение, обновленные координаты ограничивающего прямоугольника

    # Рассчитываем размеры предлагаемого расширенного (уменьшенного в масштабе) изображения
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Примечание - не используйте expand(), как new_image = filler.squeeze(1).squeeze(1).expand(3, new_h, new_w)
    # потому что все расширенные значения будут совместно использовать одну и ту же память, поэтому изменение одного пикселя изменит все

    # Поместим исходное изображение в произвольные координаты на этом новом изображении (начало координат в верхнем левом углу изображения)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Соответствующим образом отрегулируем координаты ограничивающих рамок
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):

    # Выполняет случайную обрезку способом, указанным в документе. Помогает научиться обнаруживать более крупные и неполноценные объекты.

    # :параметр image: изображение, тензор измерений (3, original_h, original_w)
    # :параметр box: ограничивающие рамки в граничных координатах, тензор измерений (n_objects, 4)
    # :параметр label: метки объектов, тензор измерений (n_objects)
    # :параметр difficulties: трудности обнаружения этих объектов, тензор измерений (n_objects)
    # :return: обрезанное изображение, обновленные координаты ограничивающего прямоугольника, обновленные метки, обновленные трудности
    original_h = image.size(1)
    original_w = image.size(2)
    # Продолжаем выбирать минимальное перекрытие до тех пор, пока не будет выполнена успешная обрезка
    while True:
        # Случайным образом нарисуем значение для минимального перекрытия
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # Если не обрезка
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Попробуем до 50 раз для этого выбора минимального перекрытия
        max_trials = 50
        for _ in range(max_trials):
            # Размеры обрезки должны быть в пределах [0.3, 1] от исходных размеров
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Соотношение сторон должно быть в [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Координаты обрезки (начало координат в верхнем левом углу изображения)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Вычисляем перекрытие Жаккарда между обрезкой и ограничивающими рамками
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects)
            overlap = overlap.squeeze(0)  # (n_objects)

            # Если ни в одном ограничивающем прямоугольнике перекрытие Жаккарда не превышает минимального значения, повторите попытку
            if overlap.max().item() < min_overlap:
                continue

            # Обрезка изображения
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Найдем центры исходных ограничивающих прямоугольников
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Найдем ограничивающие рамки, центры которых находятся в кадре
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), тензор Torch uInt8/байт может использоваться в качестве логического индекса

            # Если ни один ограничивающий прямоугольник не находится в центре кадрирования, повторим попытку
            if not centers_in_crop.any():
                continue

            # Отбросим ограничивающие рамки, которые не соответствуют этому критерию
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Вычислим новые координаты ограничивающих рамок в кадрировании
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):

    # Перевернем изображение по горизонтали.

    # :параметр image: изображение, PIL-изображение
    # :параметр box: ограничивающие рамки в граничных координатах, тензор измерений (n_objects, 4)
    # :return: перевернутое изображение, обновленные координаты ограничивающего прямоугольника

    # Перевернуть изображение
    new_image = FT.hflip(image)

    # Перевернуть бокс
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):

    # Изменим размер изображения. Для SSD 300 измените размер на (300, 300).

    # :параметр image: изображение, PIL-изображение
    # :параметров box: ограничивающие рамки в граничных координатах, тензор измерений (n_objects, 4)
    # :return: изменен размер изображения, обновлены координаты ограничивающего прямоугольника (или дробные координаты, в этом случае они остаются прежними)

    # Изменить размер изображения
    new_image = FT.resize(image, dims)

    # Изменить размер ограничивающих рамок
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # процентные координаты

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):

    # Изменяйте яркость, контрастность, насыщенность и оттенок с вероятностью 50% в случайном порядке.

    # :параметр image: изображение, PIL-изображение
    # :return: искаженное изображение

    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                adjust_factor = random.uniform(0.5, 1.5)

            new_image = d(new_image, adjust_factor)

    return new_image


def transform(image, boxes, labels, difficulties, split):

    # Применим приведенные выше преобразования.

    # :параметр image: изображение, PIL-изображение
    # :параметр box: ограничивающие рамки в граничных координатах, тензор измерений (n_objects, 4)
    # :параметр label: метки объектов, тензор измерений (n_objects)
    # :параметр difficulties: трудности обнаружения этих объектов, тензор измерений (n_objects)
    # :параметр split: один из "TRAIN" или "TEST", поскольку применяются разные наборы преобразований
    # :return: преобразованное изображение, преобразованные координаты ограничивающего прямоугольника, преобразованные метки, преобразованные трудности

    assert split in {'TRAIN', 'TEST'}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Пропустим следующие операции для оценки/тестирования
    if split == 'TRAIN':
        new_image = photometric_distort(new_image)

        # Преобразование PIL-изображения в тензор
        new_image = FT.to_tensor(new_image)

        # Увеличим изображение (уменьшите масштаб) с вероятностью 50% - полезно для тренировки обнаружения мелких объектов
        # Заполним окружающее пространство данными ImageNet, на основе которых был обучен наш базовый VGG
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Рандомная обрезка изображения
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)
        new_image = FT.to_pil_image(new_image)

        # Перевернем изображение с вероятностью 50%
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Изменим размер изображения на (300, 300) - это также преобразует абсолютные координаты границ в их дробную форму
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    new_image = FT.to_tensor(new_image)

    # Нормализуем по среднему значению и стандартному отклонению данные ImageNet, на которых был обучен наш базовый VGG
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):

    # Увеличим скорость обучения на заданный коэффициент.

    # :параметр optimizer: оптимизатор, скорость обучения которого должна быть уменьшена.
    # :параметр scale: коэффициент, на который умножается скорость обучения.

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("Снижение скоростм обучения.\n Новый LR %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):

    # Вычисляем точность top-k на основе прогнозируемых и истинных меток.

    # :параметр scores: баллы по модели
    # :параметр targets: истинные метки
    # :параметр k: k с точностью до максимума k
    # :return: максимальная точность

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer):
    
    # Сохранение модели в чекпоинт

    # :параметры epoch: epoch number
    # :параметры model: model
    # :параметры optimizer: optimizer
    
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)


class AverageMeter(object):

    # Отслеживает самые последние, средние значения, сумму и количество показателей.

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):

    # Обрезает градиенты, вычисленные во время обратного распространения, чтобы избежать резкого увеличения градиентов.

    # :оптимизатор параметров: оптимизатор с обрезаемыми градиентами
    # :параметр grad_clip: значение клипа

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                
                
def create_input_files_ic(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    
    # Создает входные файлы для обучения, проверки и тестовых данных.

    # :параметр dataset: название набора данных, одно из 'coco', 'flickr8k', 'flickr30k'
    # :параметр karpathy_json_path: путь к файлу Karpathy JSON с разделениями и подписями
    # :параметр image_folder: папка с загруженными изображениями
    # :параметр captions_per_image: количество подписей для выборки для каждого изображения
    # :параметр min_word_freq: слова, встречающиеся реже этого порога, помечаются как <ненужные> научные
    # :параметр output_folder: папка для сохранения файлов
    # :параметр max_len: не делайте выборки подписей длиннее этой длины

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Чтение Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Прочитает пути к изображениям и подписи к каждому изображению
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Обновить частоту слов
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Проверка на вменяемость
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Создать карту слов
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Создайте базовое/корневое имя для всех выходных файлов
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Сохраняет карту слов в формате JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Примеры подписей к каждому изображению, сохраните изображения в файл HDF5, а подписи и их длину - в файлы JSON
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Записывает количество подписей, которые мы отбираем для каждого изображения
            h.attrs['captions_per_image'] = captions_per_image

            # Создает набор данных внутри файла HDF5 для хранения изображений
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nСчитывание %s изображений и подписей, сохранение в файл...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Примеры подписей
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Проверка на вменяемость
                assert len(captions) == captions_per_image

                # Чтение изображений
                img = cv2.imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = cv2.resize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Сохраняет изображение в файл HDF5
                images[i] = img

                for j, c in enumerate(captions):
                    # Кодирование подписей
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Находит длину заголовка
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Проверка на вменяемость
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Сохраняет закодированные подписи и их длину в файлах JSON
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding_ic(embeddings):

    # Заполняет тензор вложения значениями из равномерного распределения.

    # :параметр embeddings: embedding tensor

    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings_ic(emb_file, word_map):

    # Создает тензор встраивания для указанной карты слов для загрузки в модель.

    # :параметр emb_file: файл, содержащий вложения (сохраненный в формате GloVe)
    # :параметр word_map: карта слов
    # :return: вложения в том же порядке, что и слова в карте слов, размер вложений

    # Находит измерение встраивания
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Создает тензор для хранения вложений, инициализируйте
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding_ic(embeddings)

    # Чтение встраиваемого файла
    print("\nЗагрузка вложений...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Игнорирование слова, если его нет в train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient_ic(optimizer, grad_clip):

    # Обрезает градиенты, вычисленные во время обратного распространения, чтобы избежать резкого увеличения градиентов.

    # :параметр optimizer: оптимизатор с обрезаемыми градиентами
    # :параметр grad_clip: значение клипа

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint_ic(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):

    # Сохраняет контрольную точку модели.

    # :параметр data_name: базовое имя обработанного набора данных
    # :параметр epoch: номер эпохи
    # :параметр epochs_since_improvement: количество эпох с момента последнего улучшения оценки BLEU-4
    # :параметр encoder: модель кодировщика
    # :параметр decoder: модель декодера
    # :параметр encoder_optimizer: оптимизатор для обновления весов кодера при точной настройке
    # :параметр decoder_optimizer: оптимизатор для обновления весов декодера
    # :параметр bleu 4: оценка validation BLEU-4 для этой эпохи
    # :параметр is_best: является ли эта контрольная точка лучшей на данный момент?

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # Если эта контрольная точка на данный момент является лучшей, сохраняет копию, чтобы она не была перезаписана худшей контрольной точкой
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):

    # Отслеживает самые последние, средние значения, сумму и количество показателей.

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate_ic(optimizer, shrink_factor):

    # Уменьшает скорость обучения на заданный коэффициент.

    # :параметр optimizer: оптимизатор, скорость обучения которого должна быть уменьшена.
    # :параметр shrink_factor: умножьте интервал (0, 1) на коэффициент, на который умножается скорость обучения.

    print("\nСнижение скорость обучения.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("Новый уровень обучения составляет %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy_ic(scores, targets, k):

    # Вычисляет максимальную точность k по прогнозируемым и истинным меткам.

    # :параметр scores: баллы по модели
    # :параметр targets: истинные метки
    # :параметр k: k с точностью до максимального значения k
    # :return: максимальная точность

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)
