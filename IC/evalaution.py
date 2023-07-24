import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import CaptionDataset
from lib import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Параметры
data_folder = 'C:/Users/Moon/Desktop/project/SSDPyTorch/JSON'  # Папка с файлами данных, сохраненными create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # Базовое имя, совместно используемое файлами данных
checkpoint = 'C:/Users/Moon/Desktop/project/SSDPyTorch/checkpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # Файл чекпоинта
word_map_file = 'C:/Users/Moon/Desktop/project/SSDPyTorch/JSON/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # Карта слов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Устанавливает устройство для тензоров модели и PyTorch
cudnn.benchmark = True  # Устанавливается в значение true только в том случае, если входные данные для модели имеют фиксированный размер; в противном случае возникают большие вычислительные затраты

# Загрузка модели
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Загрузка карты слов (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Нормализующее преобразование
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):

    # Оценивание

    # Загрузчик данных
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # Списки для хранения ссылок (истинных подписей) и гипотез (предсказаний) для каждого изображения
    # Если для n изображений у нас есть n гипотез и ссылок a, b, c ... для каждого изображения нам нужно -
    # Ссылки = [[ref1, ref1, ref1], [ref2, ref 2b], ...], гипотезы = [hyp1, hyp2, ...]
    
    references = list()
    hypotheses = list()

    # Для каждого изображения
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Перейти на графическое устройство, если оно доступно
        image = image.to(device)  # (1, 3, 256, 256)

        # Кодирование
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Сглаживание кодирования
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Мы будем рассматривать проблему как имеющую размер пакета, равный k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Тензор для хранения первых k предыдущих слов на каждом шаге; теперь это просто <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Тензор для хранения верхних k последовательностей
        seqs = k_prev_words  # (k, 1)

        # Тензор для хранения результатов лучших k последовательностей
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Списки для хранения завершенных последовательностей и оценок
        complete_seqs = list()
        complete_seqs_scores = list()

        # Начало декодирования
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Добавление
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # Для первого шага все k баллов будут иметь одинаковые баллы (начиная с некоторых предыдущих слов, h, c).
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Поиск лучших результатов и их развернутые индексы
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Преобразует развернутые индексы в фактические показатели баллов
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Добавляет новые слова в последовательности
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Отложит в сторону полные последовательности
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # Соответственно уменьшает длину луча

            # Продолжение с неполными последовательностями
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Стоп, если все продолжалось слишком долго
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # Рекомендации
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))
        references.append(img_captions)

        # Предсказание
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Подсчет очков BLEU-4
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    beam_size = 1
    print("\nОценка BLEU-4 при размере пучка %d равно %.4f." % (beam_size, evaluate(beam_size)))
