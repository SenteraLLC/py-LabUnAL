"""
This file contains (and will contain) different strategies for actively selecting superpixels.

Entropy, the most basic one is the first one which has been implemented.
"""

from collections import OrderedDict, defaultdict

import constants
import numpy as np
import torch
from dataloader import InfoLoader
from PIL import Image
from tqdm import tqdm


def turn_on_dropout(model):
    """Turn on dropout during train mode a.k.a monte carlo inference."""
    if type(model) == torch.nn.Dropout2d:
        model.train()


def monte_carlo_inference(model, num_classes, image):
    """Perform monte carlo inference to get a range of uncertain probability values to calculate the entropy later."""
    model.eval()
    model.apply(turn_on_dropout)
    # TODO: Why is it done this way and not directly passing the model output to the torch.nn.Softmax2d() function in
    #  line 38?
    softmax = torch.nn.Softmax2d()
    image = image.unsqueeze(0)
    output = torch.zeros(
        num_classes, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH
    ).type(torch.float)
    with torch.no_grad():
        for step in range(constants.MC_STEPS):
            output += softmax(model(image))[0].type(torch.float)
    output /= constants.MC_STEPS
    probabilities = output.permute(1, 2, 0).view(-1, num_classes)

    return probabilities


def select_next_batch_with_superpixels(
    model,
    training_set,
    num_classes,
    selection_count,
    dataset_name,
    superpixel_dir,
    img_size,
):
    """Perform the crux of the uncertainty measurement using entropy for actively selecting labels."""
    entropy_scores = []
    superpixel_masks = []
    image_paths = []

    """
    Detailed steps:
    1) Perform Monte Carlo dropout inference generation - done
    2) Calculate entropy of superpixels - done
    3) Pre-fetch superpixel maps - done
    4) Sort by entropy scores - done
    5) Actively select selection_count_spx num of superpixels.-Frame this logic. Take a look at other repos too.- done
    """

    images = training_set.all_train_paths
    dataset_info = InfoLoader(
        dataset_name, superpixel_dir, img_size, images, mode="val"
    )
    # image_subset_from_active_loader = training_set.get_image_subset()

    for img_idx, img_name in tqdm(enumerate(images)):

        sample = dataset_info[img_idx]
        image = sample["image"]

        # Step 1: Monte Carlo dropout inference generation
        probabilities = monte_carlo_inference(model, num_classes, image)

        # Step 2: Entropy calculation
        entropy_map = torch.zeros(
            (constants.IMAGE_WIDTH * constants.IMAGE_HEIGHT)
        ).type(torch.FloatTensor)
        for c in range(num_classes):
            # TODO: subtracted from zeros? Is that a problem? We'll see!
            # TODO: Look for articles which can visually make you understand this step
            entropy_map = entropy_map - (
                probabilities[:, c] * torch.log2(probabilities[:, c] + 1e-12)
            )
        entropy_map = (
            (entropy_map.view(constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH))
            .cpu()
            .numpy()
        )

        # TODO: Not sure if this should be done. Ask team.
        """
        idx_img_subset = image_subset_from_active_loader.index(img_name)
        mask = training_set.img_to_pixel_map[idx_img_subset]
        entropy_map[mask == 0] = 0
        """
        # Accumulate entropy scores for superpixels
        superpixels = sample["superpixel"]
        # TODO: Do we need to resize the superpixels image here? coz it's already 256, 256? Add a check, maybe?
        superpixels = np.asarray(
            Image.fromarray(superpixels).resize(
                (constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT), Image.NEAREST
            )
        )
        unique_superpixels_as_list = np.unique(superpixels).tolist()
        score_per_superpixel = defaultdict(int)
        for spx_id in unique_superpixels_as_list:
            spx_mean = entropy_map[superpixels == spx_id].mean()
            score_per_superpixel[spx_id] = spx_mean
        entropy_scores.append(score_per_superpixel)
        superpixel_masks.append(sample["superpixel"])

        image_paths.append(images[img_idx])

    # Step 3: Pre-fetch superpixel maps and other related info
    superpixel_info = []
    superpixel_scores_expanded = []
    original_image_indices = [
        training_set.all_train_paths.index(im_path) for im_path in image_paths
    ]
    for score_idx in range(len(entropy_scores)):
        superpixel_indices = list(entropy_scores[score_idx].keys())
        for (
            superpixel_idx
        ) in (
            superpixel_indices
        ):  # TODO: Check if you need information at superpixel level or just
            #  collecting all superpixels for an image is fine. Might be helpful if we need to change the
            #  uncertainty measure.
            superpixel_info.append(
                (original_image_indices[score_idx], score_idx, superpixel_idx)
            )
            superpixel_scores_expanded.append(entropy_scores[score_idx][superpixel_idx])

    # Step 4: Sort by entropy scores
    # TODO: Only the superpixel info is collected ([0]). The scores are dropped. Is it sorting by the scores? hence,
    #  x[1] ? What is the need of the first zip? What does the *sorted function return?
    _sorted_scores = np.array(
        list(
            list(
                zip(
                    *sorted(
                        zip(superpixel_info, superpixel_scores_expanded),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )
            )[0]
        )
    )
    sorted_scores = np.zeros(
        (_sorted_scores.shape[0], _sorted_scores.shape[1]), dtype=np.int32
    )
    sorted_scores[:, 0 : _sorted_scores.shape[1]] = _sorted_scores

    # Step 5: Active selection of superpixels based on sorted (in descending order) entropy scores
    selected_regions = OrderedDict()
    image_superpixels = defaultdict(list)
    total_pixels_selected = 0
    ctr = 0

    while (
        total_pixels_selected
        < selection_count * constants.IMAGE_HEIGHT * constants.IMAGE_WIDTH
        and ctr < sorted_scores.shape[0]
    ):
        # to prevent selection of the same superpixels stored over subsequent iterations
        if (
            sorted_scores[ctr, 2]
            not in training_set.image_superpixels[image_paths[sorted_scores[ctr, 1]]]
        ):
            winner_img_score_idx, winner_spx_idx = (
                sorted_scores[ctr, 1],
                sorted_scores[ctr, 2],
            )
            mask = (superpixel_masks[winner_img_score_idx] == winner_spx_idx).astype(
                np.uint8
            )
            if image_paths[winner_img_score_idx] in selected_regions:
                selected_regions[image_paths[winner_img_score_idx]] = (
                    selected_regions[image_paths[winner_img_score_idx]] | mask
                )
            else:
                selected_regions[image_paths[winner_img_score_idx]] = mask

            image_superpixels[image_paths[winner_img_score_idx]].append(winner_spx_idx)
            valid_pixels = mask.sum()
            total_pixels_selected += valid_pixels

        ctr += 1

    print(
        "Selected",
        total_pixels_selected / (constants.IMAGE_WIDTH * constants.IMAGE_HEIGHT),
        "images",
    )

    # image_subset gets updated with the new image_paths here
    training_set.expand_training_set(selected_regions, image_superpixels)
