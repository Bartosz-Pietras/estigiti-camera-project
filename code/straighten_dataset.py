import glob
import cv2 as cv
import random
from make_dataset_green import extract_green_from_img

path = r"../dataset_unchanged"
extensions = ("jpg", "jpeg")
imgs = [[], [], [], []]
for i in range(4):
    for extension in extensions:
        imgs[i].extend(glob.glob(f"{path}/stage_{i+1}/*.{extension}"))
    random.shuffle(imgs[i])
    imgs[i] = imgs[i]
    print(f"Length of imgs_{i}: {len(imgs[i])}")

counter = 0
for i in range(4):
    for img in imgs[i]:
        new_img = cv.imread(img, cv.IMREAD_UNCHANGED)
        if new_img.shape[0] < new_img.shape[1]:
            new_img = cv.rotate(new_img, cv.ROTATE_90_CLOCKWISE)
            counter += 1
        new_img_path = img.replace("dataset_unchanged", "dataset")
        # extract_green_from_img(new_img, new_img_path)
        cv.imwrite(new_img_path, new_img)

print(f"Rotated {counter} images.")
