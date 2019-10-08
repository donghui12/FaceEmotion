import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

anger_0 = []
anger_0_labels = []
disgust_1 = []
disgust_1_labels = []
fear_2 = []
fear_2_labels = []
happy_3 = []
happy_3_labels = []
sad_4 = []
sad_4_labels = []
surprised_5 = []
surprised_5_labels = []
normal_6 = []
normal_6_labels = []


def get_file(file_dir):
    for file in os.listdir(file_dir + '0'):
        anger_0.append(file_dir + '0' + '/' + file)
        anger_0_labels.append(0)
    for file in os.listdir(file_dir + '1'):
        disgust_1.append(file_dir + '1' + '/' + file)
        disgust_1_labels.append(1)
    for file in os.listdir(file_dir + '2'):
        fear_2.append(file_dir + '2' + '/' + file)
        fear_2_labels.append(2)
    for file in os.listdir(file_dir + '3'):
        happy_3.append(file_dir + '3' + '/' + file)
        happy_3_labels.append(3)
    for file in os.listdir(file_dir + '4'):
        sad_4.append(file_dir + '4' + '/' + file)
        sad_4_labels.append(4)
    for file in os.listdir(file_dir + '5'):
        surprised_5.append(file_dir + '5' + '/' + file)
        surprised_5_labels.append(5)
    for file in os.listdir(file_dir + '6'):
        normal_6.append(file_dir + '6' + '/' + file)
        normal_6_labels.append(6)

    image_list = np.hstack((anger_0, disgust_1, fear_2, happy_3, sad_4, surprised_5, normal_6))
    label_list = np.hstack((anger_0_labels, disgust_1_labels, fear_2_labels, happy_3_labels, sad_4_labels, surprised_5_labels, normal_6_labels))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    all_label_list = [int(i) for i in all_label_list]

    return all_image_list, all_label_list