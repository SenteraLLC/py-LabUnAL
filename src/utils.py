"""This file contains various utility functions for saving the selected superpixel masks, etc."""
import os

import constants
import cv2


def save_active_selections(
    dataset_name,
    runs_folder_suffix,
    selected_superpixel_masks,
    opacity_val_selected,
    opacity_val_unselected,
):
    """Make the selected superpixels appear darker in intensity than the non-selected ones."""
    directory = os.path.join(
        constants.active_runs_folder, dataset_name, runs_folder_suffix, "selections"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    for mask_name in selected_superpixel_masks:
        rgb_data = cv2.imread(
            os.path.join(
                constants.DATASET_ROOT, dataset_name, "data", "rgb", mask_name + ".jpg"
            )
        )
        b_channel, g_channel, r_channel = cv2.split(rgb_data)
        alpha_channel = (
            selected_superpixel_masks[mask_name] * opacity_val_selected
        ).astype(b_channel.dtype)
        alpha_channel[alpha_channel == 0] = opacity_val_unselected
        img_rgba = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        cv2.imwrite(os.path.join(directory, mask_name + "_mask.png"), alpha_channel)
        cv2.imwrite(os.path.join(directory, mask_name + "_overlaid.png"), img_rgba)
