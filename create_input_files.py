from lib_ic import create_input_files

if __name__ == '__main__':
    # Создание входных файлов
    create_input_files(dataset='coco',
                       karpathy_json_path='C:/Users/Moon/Desktop/project/SSDPyTorch/dataset_coco.json',
                       image_folder='D:/DATASETS',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='C:/Users/Moon/Desktop/project/SSDPyTorch',
                       max_len=50)
