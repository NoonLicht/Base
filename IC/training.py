import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from lib import *
from nltk.translate.bleu_score import corpus_bleu
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Параметры данных
data_folder = 'C:/Users/Moon/Desktop/project/SSDPyTorch/JSON' # Папка с файлами данных, сохраненными после запуска create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq' # базовое имя, совместно используемое файлами данных

# Параметры модели
emb_dim = 512 # Размерность вложений
attention_dim = 512 # Размерность слоев
decoder_dim = 512 # Размерность выпадения декодера RNN
dropout = 0.5 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Зависит от конфигурации системы
cudnn.benchmark = True # устанавливаются в значение true только в том случае, если входные данные для модели имеют фиксированный размер; в противном случае большие вычислительные затраты

# Параметры обучения
start_epoch = 0 
epochs = 120 # Количество эпох для тренировки (если не срабатывает ранняя остановка)
epochs_since_improvement = 0 # Отслеживает количество эпох с тех пор, как было улучшено качество проверки BLEU
batch_size = 1 # Размер пакетов
workers = 0 # Мультипроцессорность
encoder_lr = 1e-4 # Сорость обучения кодировщика при точной настройке
decoder_lr = 4e-4 # Скорость обучения для декодера
grad_clip = 5. # Обрезайте градиенты с абсолютным значением
alpha_c = 1. # Параметр регуляризации для "двойного стохастического внимания"
best_bleu4 = 0. # Счет BLEU-4 прямо сейчас
print_freq = 1 # Через сколько пакетов будет выводится информация в консоли
fine_tune_encoder = False # Настройка энкодера
# checkpoint = 'C:/Users/Moon/Desktop/project/SSDPyTorch/checpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' # Путь к чекпоинту, если он есть
checkpoint = None



def main():
    # Обучение и подтверждение

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Чтение карты слов
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Инициализация или загрузка чекпоинта
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Перейти на графический процессор, если он доступен
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Функция потерь
    criterion = nn.CrossEntropyLoss().to(device)

    # Пользовательские загрузчики данных
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Эпохи
    for epoch in range(start_epoch, epochs):

        # Снижение скорости обучения, если улучшения не наблюдается в течение 8 последовательных периодов, и прекращение обучения через 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate_ic(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate_ic(encoder_optimizer, 0.8)

        # Обучение одной эпохи
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # Подтверждение одной эпохи
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Проверка на улучшение
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nЭпохи с момента последнего улучшения: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Сохранение чекпоинта
        save_checkpoint_ic(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):

    # Выполняет обучение в течение одной эпохи.

    # :параметр train_loader: Загрузчик данных для обучающих данных
    # :параметр encoder: модель кодировщика
    # :параметр decoder: модель декодера
    # :параметр criterion: уровень потерь
    # :параметр encoder_optimizer: оптимизатор для обновления весов кодера (при точной настройке)
    # :параметр decoder_optimizer: оптимизатор для обновления весов декодера
    # :параметр epoch: номер эпохи
    
    decoder.train()
    encoder.train()

    batch_time = AverageMeter() # Прямое и обраное распространение
    data_time = AverageMeter() # Время чтения данных
    losses = AverageMeter() # Потери
    top5accs = AverageMeter() # TOP-5 точность

    start = time.time()

    # Пакеты
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Перейти на графический процессор, если он доступен
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Прямое распространение
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Поскольку мы расшифровали, начиная с <start>, целевыми объектами являются все слова после <start>, вплоть до <end>
        targets = caps_sorted[:, 1:]

        # Удаляет временные интервалы, которые мы не расшифровывали, или которые являются промежуточными
        # pack_padded_sequence - простой способ сделать это
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Подсчет потерь
        loss = criterion(scores, targets)

        # Добавляет двойную стохастическую регуляризацию внимания
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Обратное распространение
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Обрезка градиентов
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Обновление веса
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Показатели
        top5 = accuracy_ic(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Вывод данных
        if i % print_freq == 0:
            print('Эпоха: [{0}][{1}/{2}]\t'
                  'Время выполнения пакета {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Время загрузки данных {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Потери {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Точность {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):

    # Выполняет проверку за одну эпоху.

    # :параметр val_loader: загрузчик данных для проверки данных.
    # :параметр encoder: модель кодировщика
    # :параметр decoder: модель декодера
    # :параметр criterion: уровень потерь
    # :return: BLEU-4 балла

    decoder.eval() # Режим оценки (без отсева или пакетной нормы)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list() # Cсылки (правильные подписи) для расчета оценки BLEU-4
    hypotheses = list() # Предсказания

    with torch.no_grad():
        # Пакеты
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Перейти на графический процессор, если он доступен
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Прямое распространение
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Поскольку мы расшифровали, начиная с <start>, целевыми объектами являются все слова после <start>, вплоть до <end>
            targets = caps_sorted[:, 1:]

            # Удаляет временные интервалы, которые мы не расшифровывали, или которые являются промежуточными
            # pack_padded_sequence - простой способ сделать это
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Подсчет потерь
            loss = criterion(scores, targets)

            # Добавьте двойную стохастическую регуляризацию внимания
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Показатели
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy_ic(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Подтверждение: [{0}/{1}]\t'
                      'Время выполнения пакета {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Потери {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Точность {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Сохраняет ссылки (истинные подписи) и гипотезу (предсказание) для каждого изображения
            # Если для n изображений у нас есть n гипотез и ссылок a, b, c ... для каждого изображения нам нужно -
            # ссылки = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], гипотезы = [hyp1, hyp2, ...]

            # Параметры
            allcaps = allcaps[sort_ind] # Сортировка изображений
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps)) # удаление <start>
                references.append(img_captions)

            # Предсказание
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]]) 
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Подсчет очков BLUE-4
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * Потери - {loss.avg:.3f}, TOP-5 Точность - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
