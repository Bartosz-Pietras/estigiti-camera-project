import cv2
import numpy as np
import os


def extract_green_from_img(img, new_path):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 52, 72), (102, 255, 255))
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    cv2.imwrite(new_path, green)


def convert_dataset_to_green():
    for root, dir, filenames in os.walk("../dataset"):
        new_dir_path = f"../dataset_green/{root[20:]}"
        for filename in filenames:
            old_image_path = f"{root}/{filename}"
            new_image_path = f"{new_dir_path}/{filename}"
            img = cv2.imread(old_image_path)
            extract_green_from_img(img, new_dir_path, new_image_path)


if __name__ == "__main__":
    convert_dataset_to_green()
