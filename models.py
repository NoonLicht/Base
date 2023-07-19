import torch
from torch import nn
import torchvision
from lib_ic import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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