from torch import nn
from lib import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):

    # VGG использует базовые свертки для создания карт объектов более низкого уровня.

    def __init__(self):
        super(VGGBase, self).__init__()

        # Стандартные слови VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # шаг = 1, по умолчанию
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # Потолок (не пол) здесь для равномерного затемнения

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # Сохраняет размер, потому что шаг равен 1 (и отступ)

        # Замены для FC6 и FC7 в VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # Атрофическая свертка

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Загрузка предварительно подготовленных слои
        self.load_pretrained_layers()

    def forward(self, image):

        # Прямое распространение.

        # :параметр image: изображения, тензор размеров (N, 3, 300, 300)
        # :return: карты объектов более низкого уровня conv 4_3 и conv 7and conv7

        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), было бы 37, если бы не ceil_mode = True

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19), бассейн 5 не уменьшает размеры

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # Карты объектов более низкого уровня
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):

        # Как и в статье, мы используем VGG-16, предварительно обученный для задачи ImageNet, в качестве базовой сети.
        # В PyTorch есть один доступный, смотрите https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        # Мы копируем эти параметры в нашу сеть. Это просто для преобразования conv1 в conv5.
        # # Однако исходный VGG-16 не содержит слоев con6 и con7.
        # Следовательно, мы преобразуем fc6 и fc7 в сверточные слои и выполняем подвыборку путем прореживания. Был "уничтожен" в utils.py .

        # Текущее состояние базы
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Предварительно обученная база VGG
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Передача conv. параметры из предварительно обученной модели в текущую модель
        for i, param in enumerate(param_names[:-4]):  # исключая параметры conv6 и conv7
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Преобразуйте fc6, fc7 в сверточные слои и подвыборку (путем прореживания) до размеров conv 6 и conv 7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Примечание: слой FC размера (K), работающий на сглаженной версии (C* H*W) 2D-изображения размера (C, H, W)...
        # ...эквивалентно сверточному слою с размером ядра (H, W), входными каналами C, выходными каналами K...
        # ...работа с 2D-изображением размером (C, H, W) без заполнения

        self.load_state_dict(state_dict)

        print("\nЗагрузка базовой модели\n")


class AuxiliaryConvolutions(nn.Module):

    # Дополнительные свертки для создания карт объектов более высокого уровня.

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Вспомогательные/дополнительные витки на верхней части основания VGG
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # Шаг = 1, по умолчанию
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # уменьшение яркости, поскольку шаг > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # уменьшение яркости, поскольку шаг > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # затемнение. уменьшение, потому что заполнение = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # затемнение. уменьшение, потому что заполнение = 0

        # Инициализация параметров свертки
        self.init_conv2d()

    def init_conv2d(self):

        # Инициализация параметров свертки

        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):

        # Прямое распространение.

        # :параметр conv 7_feats: карта объектов conv 7 нижнего уровня, тензор измерений (N, 1024, 19, 19)
        # :return: карты объектов более высокого уровня conv 8_2, conv 9_2, conv 10_2 и conv 11_2

        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        # Карты объектов более высокого уровня
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):

    # Свертки для прогнозирования оценок в классе и ограничивающих рамок с использованием карт объектов более низкого и высокого уровня.
    # Ограничивающие прямоугольники (местоположения) прогнозируются как закодированные смещения относительно каждого из 8732 предыдущих (по умолчанию) прямоугольников.
    # # Смотрите 'cx cy_to_gcxgcy' в utils.py для определения кодировки.
    # Оценки класса представляют собой оценки каждого класса объектов в каждой из 8732 расположенных ограничивающих рамок.
    # Высокий балл за "фон" = нет объекта.

    def __init__(self, n_classes):
        
        # :параметр n_classes: количество различных типов объектов
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Количество предварительных блоков, которые мы рассматриваем для каждой позиции на каждой карте объектов
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        # 4 предварительных поля подразумевают, что мы используем 4 различных соотношения сторон и т.д.

        # Свертки прогнозирования локализации (прогнозирование смещений без предшествующих блоков)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        # Свертки предсказания классов (предсказывают классы в полях локализации)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

        # Инициализация параметров свертки
        self.init_conv2d()

    def init_conv2d(self):
        
        # Инициализация параметров свертки
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):

        # Прямое распространение.

        # :параметр conv 4_3_feats: карта объектов conv 4_3, тензор измерений (N, 512, 38, 38)
        # :параметр conv 7_feats: карта объектов conv 7, тензор измерений (N, 1024, 19, 19)
        # :параметр conv 8_2_feats: карта объектов conv 8_2, тензор измерений (N, 512, 10, 10)
        # :параметр conv 9_2_feats: карта объектов conv 9_2, тензор измерений (N, 256, 5, 5)
        # :параметр conv 10_2_feats: карта объектов conv 10_2, тензор измерений (N, 256, 3, 3)
        # :параметр conv 11_2_feats: карта объектов conv 11_2, тензор измерений (N, 256, 1, 1)
        # :return: 8732 местоположения и оценки класса (т.е. по сравнению с каждым предыдущим полем) для каждого изображения

        batch_size = conv4_3_feats.size(0)

        # Предсказывать границы блоков локализации (как смещения по сравнению с предыдущими блоками)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 16), чтобы соответствовать предыдущему порядку в боксе (after .view())
        # (.contiguous() гарантирует, что он хранится в непрерывном фрагменте памяти, необходимом для .view() ниже)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), на этой карте объектов в общей сложности 5776 боксов

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), на этой карте объектов в общей сложности 2116 боксов

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Предсказывать классы в полях локализации
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * n_classes), чтобы соответствовать предыдущему порядку в боксе (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), на этой карте объектов в общей сложности 5776 боксов

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), на этой карте объектов в общей сложности 2116 боксов

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # Всего 8732 боксов
        # Объединить в этом конкретном порядке (т.е. должно соответствовать порядку предыдущих полей)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
                                   dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


class SSD300(nn.Module):

    # Сеть SSD 300 - инкапсулирует базовую сеть VGG, вспомогательную и прогнозирующую свертки.

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Поскольку объекты более низкого уровня (conv 4_3_feats) имеют значительно больший масштаб, мы берем норму L2 и масштабируем заново
        # Коэффициент масштабирования изначально установлен равным 20, но определяется для каждого канала во время обратной обработки
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # в conv4_3_feats имеется 512 каналов
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):

        # Прямое распространение.

        # :параметр image: изображения, тензор размеров (N, 3, 300, 300)
        # :return: 8732 местоположения и оценки класса (т.е. по сравнению с каждым предыдущим полем) для каждого изображения

        # Запуск сверток базовой сети VGG (генераторы карт объектов нижнего уровня)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Измените масштаб conv 4_3 после нормы L2
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch автоматически передает одноэлементные измерения во время арифметики)

        # Запуск вспомогательных сверток (генераторы карт объектов более высокого уровня)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Запустите свертки прогнозирования (прогнозируйте смещения по предыдущим блокам и классам в каждом результирующем блоке локализации)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                               conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):

        # Создайте поля 8732 prior (по умолчанию) для SSD 300, как определено в документе.
        # :return: предыдущие ячейки в координатах центрального размера, тензор размеров (8732, 4)
        
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # Для соотношения сторон, равного 1, используйте дополнительный приор, масштаб которого равен среднему геометрическому значению
                        # масштаба текущей карты объектов и масштаба следующей карты объектов
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # Для последней карты объектов нет "следующей" карты объектов
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):

        # Для каждого класса выполните немаксимальное подавление (NMS) для полей, которые превышают минимальный порог.

        # :параметр predicted_locs: предсказанные местоположения/ячейки по сравнению с 8732 предыдущими ячейками, тензор измерений (N, 8732, 4)
        # :параметр predicted_scores: оценки класса для каждого из закодированных местоположений /блоков, тензор измерений (N, 8732, n_classes)
        # :параметр min_score: минимальный порог для того, чтобы поле считалось подходящим для определенного класса
        # :параметр max_overlap: максимальное перекрытие, которое могут иметь два поля, чтобы поле с более низким баллом не подавлялось с помощью NMS
        # :параметр top_k: если во всех классах много результирующих обнаружений, оставьте только верхний 'k'
        # :return: обнаружения (коробки, метки и оценки), списки длины batch_size

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Списки для хранения окончательных прогнозируемых полей, меток и оценок для всех изображений
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Расшифровка координат
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), это дробные координаты pt.

            # Списки для хранения боксов и оценок за это изображение
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Проверка для каждого класа
            for c in range(1, self.n_classes):
                # Сохранются только прогнозируемые поля и баллы, в которых баллы по этому классу превышают минимальный балл
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # тензор torch.uint8 (байт) для индексации
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Сортировка прогнозируемых полей и оценок по баллам
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Нахождение перекрытия между предсказанными блоками
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Немаксимальное подавление (NMS)

                # Тензор torch.uint8 (байт) для отслеживания того, какие предсказанные поля подавлять
                # 1 подразумевает подавление, 0 подразумевает не подавлять
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Каждый блок в порядке уменьшения баллов
                for box in range(class_decoded_locs.size(0)):
                    # Если этот флажок уже отмечен для удаления
                    if suppress[box] == 1:
                        continue

                    # Подавлять поля, перекрытия которых (с помощью этого поля) превышают максимальное перекрытие
                    # Нахождение таких полей и обновление индексов подавления
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # Операция max сохраняет ранее подавленные поля, такие как операция 'ИЛИ'

                    suppress[box] = 0

                # Храниние только неподдерживаемых полей для этого класса
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # Если объект ни в одном классе не найден, идет сохранение для 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Объединяются в отдельные тензоры
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Сохраняет только верхние k объектов
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Добавляет к спискам, в которых хранятся прогнозируемые поля и оценки для всех изображений
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # списки длины batch_size


class MultiBoxLoss(nn.Module):

    # Потеря мультибокса - функция потери для обнаружения объектов.

    # Это комбинация из:
    # (1) потеря локализации для прогнозируемых местоположений ящиков, и
    # (2) потеря достоверности прогнозируемых оценок в классе.

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()  # *smooth* L1 loss in the paper; see Remarks section in the tutorial
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):

        # Прямое распространение.

        # :параметр predicted_locs: предсказанные местоположения/ячейки по сравнению с 8732 предыдущими ячейками, тензор измерений (N, 8732, 4)
        # :параметр predicted_scores: оценки класса для каждого из закодированных местоположений /блоков, тензор измерений (N, 8732, n_classes)
        # :параметр box: истинные ограничивающие рамки объекта в граничных координатах, список из N тензоров
        # :параметр label: истинные метки объектов, список из N тензоров
        # :return: потеря мультибокса, скалярное значение

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # Для каждого предшествующего находится объект, который имеет максимальное перекрытие
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # Мы не хотим ситуации, когда объект не представлен в наших положительных (не фоновых) приоритетах -
            # 1. Объект может быть не лучшим объектом для всех приоритетов и, следовательно, не находится в object_for_each_prior.
            # 2. Все предварительные данные, относящиеся к объекту, могут быть назначены в качестве фона на основе порогового значения (0.5).

            # Чтобы исправить это -
            # Сначала найдите предыдущий, который имеет максимальное перекрытие для каждого объекта.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Затем назначим каждому объекту соответствующее значение максимального перекрытия. (Это исправляет 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # Чтобы убедиться, что эти предварительные данные соответствуют требованиям, искусственно увеличим их перекрытие более чем на 0,5. (Это исправляет 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Метки для каждого предыдущего
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Установим приоритетные значения, совпадения которых с объектами меньше порогового значения, чтобы они были фоновыми (без объекта)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Хранение
            true_classes[i] = label_for_each_prior

            # Закодируем координаты объекта центрального размера в форму, к которой мы регрессировали предсказанные поля
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Определим предварительные данные, которые являются положительными (объект /не фон)
        positive_priors = true_classes != 0  # (N, 8732)

        # Потеря локализации

        # Потеря локализации вычисляется только по положительным (не фоновым) исходным данным
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]) 

        # Примечание: индексация с помощью тензора torch.uint8 (байт) выравнивает тензор, когда индексация выполняется по нескольким измерениям (N & 8732)
        # # Итак, если predicted_locus имеет форму (N, 8732, 4), predicted_loss[positive_priors] будет иметь (всего положительных результатов, 4)

        # Потеря доверия

        # Потеря достоверности вычисляется по положительным исходным данным и наиболее сложным (hardest) отрицательным исходным данным на каждом изображении
        # То есть ДЛЯ КАЖДОГО ИЗОБРАЖЕНИЯ,
        # мы возьмем самые тяжелые (neg_pos_ratio * n_positives) отрицательные значения, т.е. те, где потери максимальны
        # Это называется Hard Negative Mining - он концентрируется на самых сильных негативах на каждом изображении, а также минимизирует дисбаланс pos / neg

        # Количество положительных и строго отрицательных исходных данных на изображение
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # Во-первых, найдем потери по всем предыдущим показателям
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # Мы уже знаем, какие предварительные данные являются положительными
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Затем найдем, какие предварительные данные являются жестко отрицательными
        # Чтобы сделать это, отсортируем только отрицательные значения в каждом изображении в порядке уменьшения потерь и возьмите верхние n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), положительные значения игнорируются (никогда в верхних n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), сортируется по убывающей твердости
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # Полная потеря

        return conf_loss + self.alpha * loc_loss


class Encoder(nn.Module):
    
    # Кодировщик

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # Предварительно обученный ImageNet ResNet-101

        # Удаляет линейные слои и слои пула
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Изменяет размер изображения до фиксированного размера, чтобы разрешить ввод изображений переменного размера
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):

        # Прямое распространение

        # :параметр images: изображения, тензор измерений (batch_size, 3, image_size, image_size)

        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):

        # Разрешить или запретить вычисление градиентов для сверточных блоков со 2 по 4 кодера.

        for p in self.resnet.parameters():
            p.requires_grad = False
        # При точной настройке выполнять только точную настройку сверточных блоков со 2 по 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):

        # :параметр encoder_dim: размер объекта в закодированных изображениях
        # :параметр decoder_dim: размер RNN декодера
        # :параметр attention_dim: size of the attention network

        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Линейный слой для преобразования закодированного изображения
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Линейный слой для преобразования выходных данных декодера
        self.full_att = nn.Linear(attention_dim, 1)  # Линейный слой для вычисления значений, подлежащих программному максимальному редактированию
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Слой softmax для вычисления веса

    def forward(self, encoder_out, decoder_hidden):

        # Прямое распространение

        # :Параметр encoder_out: закодированные изображения, тензор размерности (batch_size, num_pixels, encoder_dim)
        # :Параметр decoder_hidden: предыдущий вывод декодера, тензор размерности (batch_size, decoder_dim)

        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
        # Декодер

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):

        # :параметр attention_dim: размер сети внимания
        # :параметр embedded_dim: размер встраивания
        # :параметр decoder_dim: размер RNN декодера
        # :параметр vocab_size: размер словарного запаса
        # :параметр encoder_dim: размер объекта закодированных изображений
        # :параметр dropout: отсев

        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # встраиваемый слой
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # Декодирование LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # Линейный слой для нахождения начального скрытого состояния LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # Линейный слой для нахождения начального состояния ячейки LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # Линейный слой для создания ворот, активируемых сигмовидной мышцей
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # Линейный слой для поиска оценок по словарному запасу
        self.init_weights()  # Инициализация нескольких слоев с равномерным распределением

    def init_weights(self):

        # Инициализирует некоторые параметры значениями из равномерного распределения для облегчения сходимости.

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):

        # Загружает слой встраивания с предварительно обученными встраиваниями.

        # :параметр embeddings: предварительно обученные встраивания

        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):

        # Разрешить тонкую настройку встраиваемого слоя? (Имеет смысл запрещать только в том случае, если используются предварительно обученные встраивания).

        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):

        # Создает начальные скрытые состояния и состояния ячеек для LSTM декодера на основе закодированных изображений.

        # :параметр encoder_out: закодированные изображения, тензор размерности (batch_size, num_pixels, encoder_dim)
        # :return: hidden state, cell state

        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):

        # Прямое распространение

        # :параметр encoder_out: закодированные изображения, тензор размерности (batch_size, enc_image_size, enc_image_size, encoder_dim)
        # :параметр encoded_captions: закодированные подписи, тензор размерности (batch_size, max_caption_length)
        # :параметр caption_lengths: длина заголовка, тензор размерности (batch_size, 1)
        # :return: баллы за словарный запас, отсортированные закодированные подписи, длины декодирования, веса, индексы сортировки

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Сглаживание
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Сортировка входных данных по уменьшению длины
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Инициализировать состояние LSTM
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # Мы не будем декодировать в позиции <end>, так как мы закончили генерацию, как только сгенерировали <end>
        # Итак, длины декодирования - это фактические длины - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Создает тензоры для хранения значений предсказания слов и буквенных обозначений
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # На каждом временном шаге декодируйте с помощью 
        # взвешивания выходных данных кодера на основе предыдущих выходных данных декодера в скрытом состоянии
        # затем сгенерируйте новое слово в декодере с предыдущим словом и кодировкой, взвешенной по вниманию
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind