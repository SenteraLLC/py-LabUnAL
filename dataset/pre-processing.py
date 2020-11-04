"""This file is for splitting the filenames into train, test and validation splits."""
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

data_location = "dataset/blueriver/GreenwayAR/0629/1/data/"
splits_of_filenames_path = Path("dataset/blueriver/GreenwayAR/0629/1/filenames/")


def create_splits_of_filenames():
    """Split the data into train, test and validation and save it in filenames/ (folder)."""
    list_of_frames = [
        x.split(".")[0] for x in os.listdir(os.path.join(data_location, "rgb"))
    ]
    train_frames = [
        list_of_frames[i]
        for i in random.sample(
            range(len(list_of_frames)), int(0.60 * len(list_of_frames))
        )
    ]
    remaining_frames = [x for x in list_of_frames if x not in train_frames]
    val_frames = [
        remaining_frames[i]
        for i in random.sample(
            range(len(remaining_frames)), int(0.15 * len(list_of_frames))
        )
    ]
    test_frames = [x for x in remaining_frames if x not in val_frames]
    print(
        "train percentage:",
        len(train_frames) / len(list_of_frames),
        "\n",
        "val percentage:",
        len(val_frames) / len(list_of_frames),
        "\n",
        "test percentage:",
        len(test_frames) / len(list_of_frames),
        "\n",
    )
    with open(splits_of_filenames_path / "train_frames.txt", "w") as fptr:
        for x in train_frames:
            fptr.write(x + "\n")
    with open(splits_of_filenames_path / "val_frames.txt", "w") as fptr:
        for x in val_frames:
            fptr.write(x + "\n")
    with open(splits_of_filenames_path / "test_frames.txt", "w") as fptr:
        for x in test_frames:
            fptr.write(x + "\n")


def create_seed_set():
    """Create seed_train dataset required to begin the process of active learning."""
    train_frames = (splits_of_filenames_path / "train_frames.txt").read_text().split()
    seed_frames = [
        train_frames[i]
        for i in random.sample(range(len(train_frames)), int(0.4 * len(train_frames)))
    ]
    print("Number of seed train images:", len(seed_frames))
    with open(splits_of_filenames_path / "seed_train_frames.txt", "w") as fptr:
        for x in seed_frames:
            fptr.write(x.split(".")[0] + "\n")


def resize_images():
    """Resize contour images for the purpose of visualization."""
    for i in range(1, 21):
        image_path = (
            "../seeds-revised-master/output_contours/"
            + "IMG_"
            + "{}_contours.png".format(str(i).zfill(5))
        )
        img = Image.open(image_path)
        img_resized = img.resize((256, 256))
        img_resized.save(
            "../seeds-revised-master/output_contours_resized/"
            + "IMG_"
            + "{}.png".format(str(i).zfill(5))
        )


def rgb_to_gray():
    """And I thought the function name is self explanatory! Stupid flake8."""
    for i in range(1, 21):
        image_path = (
            data_location + "labels/" + "IMG_" + "{}.png".format(str(i).zfill(5))
        )
        img = Image.open(image_path)
        gray_img = ImageOps.grayscale(img)
        gray_img_npy = np.array(gray_img)
        gray_img_npy[gray_img_npy != 0] = 1
        gray_img = Image.fromarray(gray_img_npy)
        gray_img.save(
            data_location + "labels/" + "IMG_" + "{}.png".format(str(i).zfill(5))
        )


if __name__ == "__main__":
    # create_splits_of_filenames()
    # create_seed_set()
    resize_images()
    # rgb_to_gray()
