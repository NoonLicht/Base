import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torch_directml
from model_ssd import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from lib import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Data parameters
data_folder = 'C:/Users/Moon/Desktop/project/SSDPyTorch'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Параметры модели
n_classes = len(label_map)  # количество различных типов объектов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch_directml.device()

# Параметры обучения
checkpoint = 'checkpoint_ssd300.pth.tar' # Загрузка чекпоинта
# checkpoint = None # Если нет чекпоинта, то раскомментируй строку
batch_size = 16  # Размер пакета
iterations = 12000  # Количество итераций обучения
workers = 1  # Мультипроцессорность
print_freq = 1  # Через сколько пакетов будут обновлядтся данные в терминале
lr = 1e-2  # Скорость обучения
decay_lr_at = [8000, 10000]  # скорость обучения снижается после стольких итераций
decay_lr_to = 0.1  # снизить скорость обучения до этой доли от существующей скорости обучения
momentum = 0.9  # Импульс
weight_decay = 5e-4  # Снижение веса пакетов
grad_clip = None  # обрезайте, если градиенты увеличиваются, что может произойти при больших размерах пакета (иногда до 32) - вы узнаете об этом по ошибке сортировки при расчете потерь в MultiBox
pin_memory = False # Если True, загрузчик данных скопирует тензоры в закрепленную память устройства/CUDA, прежде чем вернуть их. Стоит ставить True только при работе через видеокарту Nvidia

cudnn.benchmark = True


def main():

    # Обучение

    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Инициализация модели или загрузка контрольной точки
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Инициализация оптимизатора с удвоенной скоростью обучения смещениям по умолчанию
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nЗагруженна контрольная точка из эпохи %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=pin_memory)  # Передача функции сопостовления

    # Вычислить общее количество эпох для обучения и эпох, в которых скорость обучения снижается (т.е. преобразовать итерации в эпохи).
    # Чтобы преобразовать итерации в эпохи, разделите итерации на количество итераций в эпоху
    # Документ обрабатывается в течение 12000 итераций с размером пакета 16, распадается после 8000 и 10000 итераций
    epochs = iterations // (len(train_dataset) // 16)
    decay_lr_at = [it // (len(train_dataset) // 16) for it in decay_lr_at]

    # Эпохи
    for epoch in range(start_epoch, epochs):

        # Скорость обучения снижается в определенные эпохи
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # Обучение эпохи
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Сохранение чекпоинта
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):

    # Тренировка одной эпохи.

    # :# :параметр train_loader: Загрузчик данных для обучающих данных
    # :параметр model: модель
    # :параметр criterion: потеря нескольких ящиков
    # :параметр optimizer: оптимизатор
    # :параметр epoch: номер эпохи

    model.train()

    batch_time = AverageMeter()  # Прямое распространение + обратное распространение, время
    data_time = AverageMeter()  # Время загрузки памяти
    losses = AverageMeter()  # Потери

    start = time.time()

    # Пакеты
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Прямое распространение
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Потери
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        # Обратное распространение
        optimizer.zero_grad()
        loss.backward()

        # Обрезайте градиенты, если это необходимо
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Обновление модели
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Вывод данных
        if i % print_freq == 0:
            print('Эпоха: [{0}][{1}/{2}]\t'
                  'Время выполнения пакета {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Время передачи данных {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Потери {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels


if __name__ == '__main__':
    main()