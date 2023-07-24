from torchvision import transforms
# import torch_directml
from lib import *
from PIL import Image, ImageDraw, ImageFont
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Выбор девайса, если есть карточка от NVIDIA, то все вычисления будут проводится на ней
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch_directml.device()

# Загрузка чекпоинта
checkpoint = 'C:/Users/Moon/Desktop/project/SSDPyTorch/checkpoints/checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nЗагруженна контрольная точка из эпохи %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Преобразование фотографий 
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    
    # Обнаруживайте объекты на изображении с помощью обученного SSD 300 и визуализируйте результаты.
    # :параметр original_image: изображение, PIL-изображение
    # :параметр min_score: минимальный порог для того, чтобы обнаруженный блок считался соответствующим определенному классу
    # :параметр max_overlap: максимальное перекрытие, которое могут иметь два блока, чтобы тот, у которого более низкий балл, не подавлялся с помощью немаксимального подавления (NMS)
    # :параметр top_k: если во всех классах много результирующих обнаружений, оставьте только верхний 'k'
    # :параметр suppress: классы, которые, как вы точно знаете, не могут быть в изображении или которые вы не хотите видеть в изображении, список
    # :return: аннотированное изображение, PIL-изображение

    # Преобразование
    image = normalize(to_tensor(resize(original_image)))

    # Перейти к устройству по умолчанию
    image = image.to(device)

    # Прямое распространение
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Обнаружение объектов на выходе 
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # перекидывание обнаружения боксов на процессор
    det_boxes = det_boxes[0].to('cpu')

    # Преобразование в исходные размеры изображения
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Декодирование целочисленных меток классов на мозностях процессора
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # Если объекты не найдены, обнаруженным меткам будет присвоено значение ['0.'], т.е. ['background'] в SSD300.detect_objects() в model.py
    if det_labels == ['background']:
        # Возврат исходного изображения
        return original_image

    # Аннотации
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Подавление классов 
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Боксы
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # Можно дублировать и увеличить толщину рамки

        # Текст
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    img_path = 'C:/Users/Moon/Desktop/project/SSDPyTorch/2007/JPEGImages/001334.jpg' # Путь к изображению (любое), которое и прогоняем через нейронку
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()