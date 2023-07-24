from lib import create_input_files_ic

if __name__ == '__main__':
    # Создание входных файлов
    create_input_files_ic(dataset='coco',
                       karpathy_json_path='C:/Users/Moon/Desktop/project/SSDPyTorch/JSON/dataset_coco.json',
                       image_folder='D:/DATASETS',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='C:/Users/Moon/Desktop/project/SSDPyTorch/JSON',
                       max_len=50)
