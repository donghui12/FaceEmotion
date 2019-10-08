import csv
import os
from PIL import Image
import numpy as np

DATABASE_PATH = 'F:\\Python\\10.8\\FaceEmotion\\datasets\\fer2013'  # 数据集路径
DATASETS_PATH = 'F:\\Python\\10.8\\FaceEmotion\\datasets\\fer2013'

csv_file = os.path.join(DATASETS_PATH, 'fer2013.csv')
train_csv = os.path.join(DATASETS_PATH, 'train.csv')
val_csv = os.path.join(DATASETS_PATH, 'val.csv')
test_csv = os.path.join(DATASETS_PATH, 'test.csv')

train_set = os.path.join(DATASETS_PATH, 'train')
val_set = os.path.join(DATASETS_PATH, 'val')
test_set = os.path.join(DATASETS_PATH, 'test')


def load_file():
    """
    对数据集进行分类，得到train.csv, test.csv, val.csv
    :return:
    """
    with open(csv_file) as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        rows = [row for row in csv_reader]

        train = [row[:-1] for row in rows if row[-1] == 'Training']
        csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[: -1]] + train)
        print('训练数据集写入成功！其长度为：', len(train))

        val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
        csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[: -1]] + val)
        print('确认数据集写入成功！其长度为：', len(val))

        test = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[: -1]] + test)
        print('测试数据集写入成功！其长度为：', len(test))


def restore_to_img():
    """
    根据特征值将其还原为48*48的灰度图像
    :return:
    """
    for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        num = 1
        with open(csv_file) as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            for i, (label, pixel) in enumerate(csv_reader):
                pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
                subfolder = os.path.join(save_path, label)
                if not os.path.exists(subfolder):
                    os.mkdir(subfolder)
                im = Image.fromarray(pixel).convert('L')
                image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
                print(image_name)
                im.save(image_name)


def main():
    # 执行顺序,先执行load_file() ,把restore_to_img()注释掉，然后执行restore_to_img(),将load_file()注释掉
    # load_file()
    # restore_to_img()
    pass


main()
