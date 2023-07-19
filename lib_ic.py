import os
import numpy as np
import h5py
import json
import torch
import cv2
from tqdm import tqdm
from collections import Counter
from random import *


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
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


def init_embedding(embeddings):

    # Заполняет тензор вложения значениями из равномерного распределения.

    # :параметр embeddings: embedding tensor

    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):

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
    init_embedding(embeddings)

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


def clip_gradient(optimizer, grad_clip):

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


def accuracy(scores, targets, k):

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
