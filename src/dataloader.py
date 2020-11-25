"""
This file will contain following functionalities.

1) Applying custom transforms to images
2) Dataloader class and some basic functionalities like get_labeled_pixel_count(),
   get_fraction_of_labeled_data(), expand_training_set(), load_selections(), get_selections().
   All other functionalities related to data loading to be developed in the future can be made part
   of this dataloader class.

"""

import os
from collections import OrderedDict, defaultdict

import constants
import image_transforms
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm


class ActiveDataLoader(Dataset):
    """This class contains the implementation of a dataloader suited for Active Learning based functionalities."""

    def __init__(self, dataset_name, superpixel_dir, img_size, mode):
        """
        Set the values of the name of the dataset, the location of the superpixel segmentations, image size.

        Also set the mode (train/test/validate).
        """
        self.dataset_name = dataset_name
        self.all_train_paths = None
        self.superpixel_dir = superpixel_dir
        self.transform = image_transforms.image_transforms_blueriver
        self.mode = mode

        if mode == "train":
            with open(
                os.path.join(
                    constants.DATASET_ROOT,
                    self.dataset_name,
                    "filenames",
                    "train_frames.txt",
                ),
                "r",
            ) as file_ptr:
                self.all_train_paths = [
                    "{}".format(x.strip()) for x in file_ptr.readlines() if x != ""
                ]

            # TODO: Do I need to take image_subset, img_to_pixel_map, image_superpixels out from the if "train"
            #  condition?
            self.image_subset = []

            # TODO: Cross-check if this is loaded only once, i.e image_subset gets populated only for the first time
            #  from seed_train_frames.txt and only updated after that.
            with open(
                os.path.join(
                    constants.DATASET_ROOT,
                    self.dataset_name,
                    "filenames",
                    "seed_train_frames.txt",
                ),
                "r",
            ) as fptr:
                self.image_subset = [
                    "{}".format(x.strip()) for x in fptr.readlines() if x != ""
                ]

            self.img_to_pixel_map = OrderedDict({})
            for path in self.image_subset:
                self.img_to_pixel_map[path] = np.ones(img_size, dtype=np.uint8)

            self.image_superpixels = defaultdict(list)
            for i in range(len(self.image_subset)):
                superpixel_path = os.path.join(
                    constants.DATASET_ROOT,
                    self.dataset_name,
                    "data",
                    self.superpixel_dir,
                    "{}.png".format(self.image_subset[i]),
                )
                self.image_superpixels[self.image_subset[i]] = np.unique(
                    np.asarray(imread(superpixel_path), dtype=np.int32)
                ).tolist()

        if mode == "test" or mode == "val":
            with open(
                os.path.join(
                    constants.DATASET_ROOT,
                    self.dataset_name,
                    "filenames",
                    "{}".format(self.mode) + "_frames.txt",
                ),
                "r",
            ) as file_ptr:
                self.image_subset = [
                    "{}".format(x) for x in file_ptr.readlines() if x != ""
                ]

    def get_labeled_pixel_count(self):
        """
        Get the no. of pixels which are labeled in a particular iteration. The value is cumulative over iterations.

        Helper for get_fraction_of_labeled_data().
        """
        pixel_count = 0
        for img in self.img_to_pixel_map.keys():
            pixel_count += self.img_to_pixel_map[img].sum()
        return pixel_count

    def get_fraction_of_labeled_data(self):
        """Get the fraction of the data which is labeled. Uses get_labeled_pixel_count as helper."""
        return self.get_labeled_pixel_count() / (
            len(self.all_train_paths) * constants.IMAGE_HEIGHT * constants.IMAGE_WIDTH
        )

    def expand_training_set(self, selected_regions, image_superpixels):
        """Use in train_active.py to add the actively selected new labels to the existing dataset."""
        for image in image_superpixels:
            self.image_superpixels[image].extend(image_superpixels[image])

        for img in selected_regions:
            if img in self.img_to_pixel_map:
                self.img_to_pixel_map[img] = (
                    self.img_to_pixel_map[img] | selected_regions[img]
                )

            else:
                self.img_to_pixel_map[img] = selected_regions[img]
                self.image_subset.append(img)

    def load_selections(self, active_selections_path):
        """
        Only called for loading active selections saved during previous active iterations in train_active.py.

        Hence the creation of the new dictionary.
        """
        self.img_to_pixel_map = OrderedDict({})
        self.image_subset = []
        for img in tqdm(
            os.listdir(active_selections_path),
            desc="Loading uncertain superpixels from previous active runs...",
        ):
            active_spx_img = img.split(".")[0]
            self.img_to_pixel_map[active_spx_img] = imread(
                os.path.join(active_selections_path, img)
            )
            self.image_subset.append(active_spx_img)

    def get_selections(self):
        """Call in train_active.py to save the superpixel selections."""
        return self.img_to_pixel_map

    def get_image_subset(self):
        """Return the training dataset for the current active iteration."""
        return self.image_subset

    def reset_dataset(self):
        """Reset the dataset just before inference in train_active.py."""
        self.image_subset = self.image_subset[: self.len_image_subset]

    def fix_list_multiple_of_batch_size(self, paths, batch_size):
        """Make the train dataset a multiple of the batch.

        This is done because new added images may not be divisible by batch_size. Helper for below function.
        """
        remainder = len(paths) % batch_size
        if remainder != 0:
            num_new_entries = batch_size - remainder
            new_entries = paths[:num_new_entries]
            paths.extend(new_entries)
        return paths

    def make_dataset_multiple_of_batchsize(self, batch_size):
        """Make the train dataset a multiple of the batch_size."""
        self.len_image_subset = len(self.image_subset)  # TODO: fix this style error
        self.image_subset = self.fix_list_multiple_of_batch_size(
            self.image_subset, batch_size
        )

    def __getitem__(self, index):
        """Given the index of an image, read the image, label and superpixel. Pass through needed transforms."""
        img_name = self.image_subset[index]

        rgb_img = imread(
            os.path.join(
                constants.DATASET_ROOT,
                self.dataset_name,
                "data",
                "rgb",
                "{}.jpg".format(img_name),
            )
        )
        label_img = imread(
            os.path.join(
                constants.DATASET_ROOT,
                self.dataset_name,
                "data",
                "labels",
                "{}.png".format(img_name),
            )
        )

        spx_img = self.image_superpixels[index]

        if self.transform:
            sample = self.transform(rgb_img, label_img, self.mode)

        # This step is so important to consider only the labels of the superpixels which have been added to the current
        # active dataset and will go for the next "active" training. While calculating the loss in trainer.py,
        # in torch.nn.CrossEntropyLoss, ignore_index is assigned a value of 255 to mask out the pixels which have
        # not been marked for labeling yet.

        # TODO: Do we need to add the ignore_index immediately after we expand the training dataset, coz in the
        #  second and subsequent selection rounds when we would be reading label images which have been partially
        #  labeled, we don't know what would be read for superpixels that have not been labeled. Check out other repos.
        #  It does not pose a problem here because we are reading fully labeled images.
        mask = self.img_to_pixel_map[img_name]
        label_img_transformed = sample["label"]
        label_img_transformed[mask == 0] = 255
        sample["label"] = label_img_transformed

        if self.mode == "train":
            sample["superpixel"] = spx_img

        return sample

    def __len__(self):
        """Give the length of the dataset for the current active iteration."""
        return len(self.image_subset)


class InfoLoader(Dataset):
    """This class is only for loading the rgb, label and superpixel images and passing through selected transforms.

    In the future we can also load other information pertaining to an image in this class.
    """

    def __init__(self, dataset_name, superpixel_dir, img_size, image_paths, mode="val"):
        """Initialize to validation transforms by passing in the mode, beside other parameters."""
        self.dataset_name = dataset_name
        self.all_train_paths = None
        self.superpixel_dir = superpixel_dir
        self.transform = image_transforms.image_transforms_blueriver
        self.mode = mode
        self.image_paths = image_paths
        self.img_size = img_size

    def __getitem__(self, index):
        """Given the index of an image, read the image, label & superpixels. Pass through "selected" transforms only."""
        img_name = self.image_paths[index]
        rgb_img = imread(
            os.path.join(
                constants.DATASET_ROOT,
                self.dataset_name,
                "data",
                "rgb",
                "{}.jpg".format(img_name),
            )
        )
        label_img = imread(
            os.path.join(
                constants.DATASET_ROOT,
                self.dataset_name,
                "data",
                "labels",
                "{}.png".format(img_name),
            )
        )

        spx_img = imread(
            os.path.join(
                constants.DATASET_ROOT,
                self.dataset_name,
                "data",
                self.superpixel_dir,
                "{}.png".format(img_name),
            )
        )
        if self.transform:
            sample = self.transform(rgb_img, label_img, self.mode)

        sample["superpixel"] = spx_img

        return sample
