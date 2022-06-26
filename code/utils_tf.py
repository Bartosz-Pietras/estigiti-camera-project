import glob
import os
import shutil
from typing import List

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


def use_gpu() -> None:
    """Search for available devices and use GPU if possible."""
    devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(devices)}")
    tf.config.experimental.set_memory_growth(devices[0], True)


def plot_confusion_matrix(cf_matrix: np.ndarray, classes: List) -> None:
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n')
    ax.set_xlabel('Predicted stage')
    ax.set_ylabel('Actual stage\n')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    plt.show()


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Based on predictions and actual labels, return the confusion matrix."""
    return confusion_matrix(y_true=y_true, y_pred=y_pred)


def save_model(model: keras.Sequential, name: str) -> None:
    """Given the name and the model, save the entire model to the 'model' directory."""
    if os.path.isfile(f"model/{name}.h5") is False:
        model.save(f"model/{name}.h5")


def save_weights(model: keras.Sequential, name: str) -> None:
    """Given the name and the model, save its weights to the 'weights' directory."""
    if os.path.isfile(f'weights/{name}.h5') is False:
        model.save_weights(f"weights/{name}.h5")


def get_num_of_stages(path: str) -> int:
    """Go over subdirectories in a directory and get how many of them are stage directories."""
    return len(glob.glob(f"{path}/stage*"))


def get_images(path: str) -> List:
    """Get the shuffled list of images in a given directory."""
    return shuffle(glob.glob(f"{path}/*.jpg") + glob.glob(f"{path}/*.jpeg"))


def get_stages_list(num_stages: int) -> List:
    """Return a list with the numbered stage names."""
    stages = []
    for i in range(num_stages):
        stages.append(f"stage_{i + 1}")

    return stages

from PIL import Image
import numpy as np
from skimage import transform

def load(image):
   np_image = np.array(image).astype('float32')/255
   np_image = transform.resize(np_image, (298, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def predict_image(image, model):
    img = load(image)
    output = model.predict(img)
    print(output)
    max_value = np.amax(output)
    max_index = np.where(output == max_value)[1][0]
    return (max_index, max_value)



def split_dataset(path: str, train_part: float = 0.7, valid_part: float = 0.2) -> None:
    """
    This function splits the dataset into 3 subdirectories:
    - train
    - valid
    - test

    The proportion between them is given by 2 parameters:
    - train_part (0.7 by default)
    - valid_part (0.2 by default)

    The test part consists of the rest of the unclassified images, so 0.1 by default.
    """
    # Check how many stages the dataset is divided into and get a list of those stages.
    directories = ("train", "valid", "test")
    stages = get_stages_list(get_num_of_stages(path))

    # Create a dict that stores lists of all images in all stages.
    images_per_stage = {}
    for stage in stages:
        images_per_stage[stage] = get_images(f"{path}/{stage}")

    # Create train, valid and test directories and stage subdirectories inside them.
    for directory in directories:
        dir_path = f"{path}/{directory}"
        os.makedirs(dir_path)

        for stage in stages:
            os.makedirs(f"{dir_path}/{stage}")

    # Loop through the newly created directories, determine which images are going to be in which directory
    # and subdirectory and copy those images there.
    for directory in directories:
        for stage in stages:
            num_images = len(images_per_stage[stage])
            train_imgs = int(num_images * train_part)
            valid_imgs = int(num_images * valid_part)

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


def check_directories(path: str) -> bool:
    """This function checks whether directories train, valid and test exist."""
    return os.path.isdir(f"{path}/train") and os.path.isdir(f"{path}/valid") and os.path.isdir(f"{path}/test")


def remove_directories(path: str) -> None:
    """Removes recursively directories 'train', 'valid' and 'test'."""
    shutil.rmtree(f"{path}/train")
    shutil.rmtree(f"{path}/valid")
    shutil.rmtree(f"{path}/test")

if __name__ == "__main__":
    # image = cv2.imread('../dataset/stage_1/20220517_161258.jpg', 0)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    model = tf.keras.models.load_model("../model/model_dataset_equal_SGD_lr_1e-05_kernel_size_3")
    print(model.summary())

    # Saving the Model in H5 Format
    tf.keras.models.save_model(model, "../model/SGD_lr_1e-05_kernel_size_3.h5")
