import os
import shutil
from onedrivedownloader import download

ROOT_PATH = "../"
DATASET_PATH = ROOT_PATH + "dataset"
URL = r"https://politechnikawroclawska-my.sharepoint.com/:u:/g/pers" \
      "onal/253133_student_pwr_edu_pl/ESr4b-6X0GVIjWJhRXFRU0ABGFH_4gIRqrpyzQqchuHL1A?e=B1xx6P"


def check_dataset():
    """Checks if a directory with a dataset exists."""
    return os.path.isdir(DATASET_PATH)


def handle_dataset_directory():
    """If a dataset directory exists, it deletes it, creates a new one and downloads the latest dataset to it."""
    if check_dataset():
        shutil.rmtree(DATASET_PATH)
    download(url=URL, filename="dataset.zip", unzip=True, unzip_path=ROOT_PATH)
    os.remove(f"{os.getcwd()}/dataset.zip")


if __name__ == "__main__":
    handle_dataset_directory()
