### Скачайте датасет [1](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar), [2](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), [3](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) и объедините за 2007 год в папку 2007, а за 2012 в 2012 соответственно.

### Вот еще 2 датасета: [1](http://images.cocodataset.org/zips/train2014.zip) и [2](http://images.cocodataset.org/zips/val2014.zip)

### Эти [жисоны](https://mega.nz/file/gtEiTAjQ#jgtEGh9JWjJ5tsRYmjSeGZqhL64SvRA4s0VXLaM5wPk) закинуть в папку с программой

### Скачайте [checkpoint для SSD](https://mega.nz/file/x09iGLYB#U-nxmoh4-x5K2Ftq-XKjB1WwgW8fS1fynjSDcWjkU88) и [checkpoint для IC](https://mega.nz/folder/RtNwHDha#Cl5piynrtTCavAACjUziLw). 
Его надо еще обучать, тк до сих пор в некотрых местах работает некоректно.

### Для того, чтоб все работало вам надо установить [PyTorch](https://pytorch.org/get-started/locally/) под свой конфиг (но нужно выбрать CUDA 11.8 для удобства).

![image](https://github.com/NoonLicht/base/assets/121355541/25a12112-f2ff-4df9-9e87-d90d94e61ad7)

Не забудьте еще установить CUDAToolKit 
```
pip install cudatoolkit=11.8
pip install opencv-python
pip install Pillow==9.5.0
```
Еще вам надо будет скачать [cuDNN](https://developer.nvidia.com/cudnn) тоже под CUDA 11.8.
