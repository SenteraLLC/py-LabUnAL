"""This file is used for actively training the model."""

# TODO: 1) take out train_active.py from src and accordingly modify the imports
#      2) change the skimage imports to PIL images
#      3) change the format of the dataset_name

#  steps prior to active_selection.py
#       1) create a deeplab model
#       2) Create a trainer class - visualization needs to be done using TB
#       3) Define a scheduler like SGD, Adam
#       4) Run a loop for say 5 epochs for the training. For now, perform only training. Do not work on the validation.
#       5) Reset the dataset, i.e. image_subset now contains only seeds no. of images for first run and actively
#          selected images out of the whole training lot for subsequent runs. This is done because
#          make_dataset_multiple_of_batchsize is called earlier which appends extra images to make current training
#          dataset a multiple of batch size.
#       6) Perform inference
#       7) Implement select_next_batch_with_superpixels() - done

import os

import utils
from active_selection import select_next_batch_with_superpixels
from dataloader import ActiveDataLoader
from model.deeplab import DeepLab
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer import Trainer


def main():
    """Perform the active training here."""
    # TODO: Put these args in a CLICK module
    # params
    dataset_name = "blueriver/GreenwayAR/0629/1"
    superpixel_dir = "superpixels/"
    img_size = (256, 256)
    mode = "train"
    active_iterations = 5
    num_classes = 2
    batch_size = 2
    learning_rate = 0.001
    epochs = 5
    runs_folder = "./runs"
    checkpoint = "./checkpoint"
    selection_count = 1  # how many images are going to be actively selected?, this number is for our small dataset.
    opacity_val_selected = 170
    opacity_val_unselected = 200

    if dataset_name == "blueriver/GreenwayAR/0629/1":
        train_dataset = ActiveDataLoader(dataset_name, superpixel_dir, img_size, mode)
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=0
        )
        print("Loaded a dataset with {} images".format(len(train_dataset)))
    # Just a check
    pbar = tqdm(train_loader)
    for i, sample in enumerate(pbar):
        print("Checking if loaded, image:", i)

    for selection_iter in range(active_iterations):
        fraction_of_data_labeled = int(
            round(train_dataset.get_fraction_of_labeled_data() * 100)
        )

        # Load previously selected superpixels continuing from a previous active iteration
        if os.path.exists(
            os.path.join(
                runs_folder,
                dataset_name,
                checkpoint,
                "runs_{}".format(fraction_of_data_labeled),
                "selections",
            )
        ):
            train_dataset.load_selections(
                os.path.join(
                    runs_folder,
                    dataset_name,
                    checkpoint,
                    "runs_{}".format(fraction_of_data_labeled),
                    "selections",
                )
            )
        else:
            print(
                "Now prepping to train the model for active iteration", selection_iter
            )

            train_dataset.make_dataset_multiple_of_batchsize(batch_size)

            model = DeepLab(
                num_classes=num_classes,
                backbone="mobilenet",
                output_stride=16,
                sync_bn=False,
                mc_dropout=True,
            )
            trainer = Trainer(model, train_dataset, batch_size, learning_rate)

            # now train, skipping validation for now
            lr_scheduler = trainer.lr_scheduler
            for epoch in range(epochs):
                print("Epoch, Active Iteration:", epoch, ", ", selection_iter)
                trainer.training(epoch)
                if lr_scheduler:
                    lr_scheduler.step()

            train_dataset.reset_dataset()

            # TODO: reset and then perform inference here
            select_next_batch_with_superpixels(
                model,
                train_dataset,
                num_classes,
                selection_count,
                dataset_name,
                superpixel_dir,
                img_size,
            )

            fraction_of_data_labeled_after_selection = int(
                round(train_dataset.get_fraction_of_labeled_data() * 100)
            )
            # saving the superpixel masks as images
            utils.save_active_selections(
                dataset_name,
                "runs_{}".format(fraction_of_data_labeled_after_selection),
                train_dataset.get_selections(),
                opacity_val_selected,
                opacity_val_unselected,
            )

        print(selection_iter, " / Train-set length: ", len(train_dataset))


if __name__ == "__main__":
    main()
