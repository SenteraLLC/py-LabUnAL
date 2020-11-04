"""
This separate file contains custom transforms suited for different datasets.

All transforms for a particular dataset are instantiated in a separate function.
"""
import random

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.

    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        """Initialize mean and std. dev for the image being normalized."""
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """Implement Normalize transform."""
        img = sample["image"]
        label = sample["label"]
        img = img.astype(np.float32)
        label = label.astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {"image": img, "label": label}


class ToTensor(object):
    """This function converts nd-arrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Swap color axis because.

        numpy image: H x W x C
        torch image: C X H X W.

        """
        img = sample["image"]
        label = sample["label"]

        img = img.astype(np.float32)

        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))

        label = label.astype(np.float32)

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()

        return {"image": img, "label": label}


class RandomGaussianBlur(object):
    """This function adds gaussian noise to the image and labels."""

    def __call__(self, sample):
        """Decide whether to opt for RandomGaussianBlur() based on output of a random number generator, random()."""
        img = sample["image"]
        label = sample["label"]
        if random.random() < 0.5:
            img = gaussian_filter(img, sigma=random.random())

        return {"image": img, "label": label}


class RandomHorizontalFlip(object):
    """This function flips the label and image."""

    def __call__(self, sample):
        """Flips the label and image based on output of a random number generator, random()."""
        img = sample["image"]
        label = sample["label"]

        if random.random() < 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)

        return {"image": img, "label": label}


class ResizeImage(object):
    """This transform resizes the image and label based on the input size provided.

    Args:
        size(tuple) : size of the image to be resized to.

        Currently not instantiated due to issues in resizing using PIL.
    """

    def __call__(self, sample, size=(256, 256)):
        """Implement ResizeImage."""
        self.size = size
        img = sample["image"]
        label = sample["label"]
        w, h = img.shape[1], img.shape[0]

        if w > h:
            oh = self.size[0]
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.size[1]
            oh = int(1.0 * h * ow / w)

        img = np.array(Image.fromarray(img).resize(ow, oh))
        label = np.array(Image.fromarray(label).resize(ow, oh), Image.NEAREST)

        return {"image": img, "label": label}


def image_transforms_blueriver(image, label, mode):
    """Perform instantiation of transforms for the blueriver dataset."""
    if mode == "train":
        composed_transforms = transforms.Compose(
            [
                RandomHorizontalFlip(),
                RandomGaussianBlur(),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # one for each channel
                # ResizeImage(),
                ToTensor(),
            ]
        )

        return composed_transforms({"image": image, "label": label})

    elif mode == "test" or mode == "val":
        composed_transforms = transforms.Compose(
            [
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # ResizeImage(),
                ToTensor(),
            ]
        )

        return composed_transforms({"image": image, "label": label})
