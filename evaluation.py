from lib import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Форматирование при печати
pp = PrettyPrinter()

# Параметры запуска оценивания
data_folder = './' # Корневаяя папка с программой
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1 # Размер пакетов (лучше ставить от 16 до 64)
workers = 1 # Мультипроцессорность (запускает несколько процессов для ускорения вычисления, количество ядер = количество workers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoint_ssd300.pth.tar' # Путь к чекпоинту
pin_memory = False # Если True, загрузчик данных скопирует тензоры в закрепленную память устройства/CUDA, прежде чем вернуть их. Стоит ставить True только при работе через видеокарту Nvidia

# Загрузка чекпоинта 
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Запуск теста
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=pin_memory)

def evaluate(test_loader, model):
    model.eval()

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    with torch.no_grad():
        # Пакеты
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Прямое распространение
            predicted_locs, predicted_scores = model(images)

            # Обнаружение объектов
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)

            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
        # Вычисление средней точность
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    pp.pprint(APs)
    print('\nСредняя точность (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)
