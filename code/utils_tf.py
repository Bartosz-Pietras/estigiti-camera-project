import numpy as np
from random import randint
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import itertools
import shutil
import matplotlib.pyplot as plt
import os
import glob


def plot_confusion_matrix(cf_matrix):
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n')
    ax.set_xlabel('Predicted stage')
    ax.set_ylabel('Actual stage\n')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Sick', 'Not sick'])
    ax.yaxis.set_ticklabels(['Sick', 'Not sick'])

    plt.show()


def create_dataset(total_samples):
    regular_samples = int(0.95*total_samples)
    abnormal_samples = total_samples - regular_samples

    labels = []
    samples = []

    for i in range(abnormal_samples):
        random_younger = randint(13, 64)
        samples.append(random_younger)
        labels.append(1)

        random_older = randint(65, 100)
        samples.append(random_older)
        labels.append(0)

    for i in range(regular_samples):
        random_younger = randint(13, 64)
        samples.append(random_younger)
        labels.append(0)

        random_older = randint(65, 100)
        samples.append(random_older)
        labels.append(1)

    return np.array(samples), np.array(labels)


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)


def save_model(model, name: str):
    if os.path.isfile(f"model/{name}.h5") is False:
        model.save(f"model/{name}.h5")


def save_weights(model, name: str):
    if os.path.isfile(f'weights/{name}.h5') is False:
        model.save_weights(f"weights/{name}.h5")


def get_num_of_stages(path: str):
    return len(next(os.walk(path))[1])


def get_images(path: str):
    return shuffle(glob.glob(f"{path}/*.jpg") + glob.glob(f"{path}/*.jpeg"))


def split_dataset(path: str = r"../dataset"):
    num_stages = get_num_of_stages(path)
    directories = ("train", "valid", "test")
    stages = []

    for i in range(num_stages):
        stages.append(f"stage_{i+1}")

    images_per_stage = {}

    for stage in stages:
        images_per_stage[stage] = get_images(f"{path}/{stage}")

    for directory in directories:
        dir_path = f"{path}/{directory}"
        os.makedirs(dir_path)

        for stage in stages:
            os.makedirs(f"{dir_path}/{stage}")

    for directory in directories:
        for stage in stages:
            num_images = len(images_per_stage[stage])
            train_imgs = int(num_images * 0.7)
            valid_imgs = int(num_images * 0.2)

            if directory == "train":
                for img in images_per_stage[stage][:train_imgs]:
                    pure_img = img.split("\\")[-1]
                    shutil.copy(img, f"{path}/{directory}/{stage}/{pure_img}")

            elif directory == "valid":
                for img in images_per_stage[stage][train_imgs:train_imgs+valid_imgs]:
                    pure_img = img.split("\\")[-1]
                    shutil.copy(img, f"{path}/{directory}/{stage}/{pure_img}")

            elif directory == "test":
                for img in images_per_stage[stage][train_imgs+valid_imgs:]:
                    pure_img = img.split("\\")[-1]
                    shutil.copy(img, f"{path}/{directory}/{stage}/{pure_img}")








