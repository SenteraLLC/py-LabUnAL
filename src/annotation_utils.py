import copy
import json

from skimage.io import imread, imsave


def generate_json_for_label(superpixels, label_or_not, rgb):
    # TODO: Refactor this to remove the rgb parameter from the function because we won't be overlaying the superpixels
    #         on the image.
    coordinates_dict_list = []
    rows = rgb.shape[0]
    cols = rgb.shape[1]
    rgb_copy = copy.deepcopy(rgb)
    for i in range(rows):
        for j in range(cols):

            # gather contour information of superpixels only if they have been selected by the AL algo to be labeled
            if label_or_not[i][j] > 0:
                label = superpixels[i][j]

                label_on_top = label
                if i > 0:
                    label_on_top = superpixels[i - 1][j]

                label_on_bottom = label
                if i < rows - 1:
                    label_on_bottom = superpixels[i + 1][j]

                label_on_left = label
                if j > 0:
                    label_on_left = superpixels[i][j - 1]

                label_on_right = label
                if j < cols - 1:
                    label_on_right = superpixels[i][j + 1]

                if (
                    label != label_on_top
                    or label != label_on_bottom
                    or label != label_on_left
                    or label != label_on_right
                ):
                    coordinates_dict = {"x": j, "y": i}
                    coordinates_dict_list.append(coordinates_dict)
                    rgb_copy[i][j] = 255

    imsave("selected_superpixels_only.png", rgb_copy)

    with open("contour_coordinates.json", "w+") as output_file:
        json.dump(coordinates_dict_list, output_file)

    # just a check to see if the json was saved properly
    with open("contour_coordinates.json") as f:
        data = json.load(f)


def initiate_labeling_job(image_subset, img_to_pixel_map, image_superpixels):
    """
    These are the steps to initiate a labeling job
    1) For the images, plot the superpixels in an image, gray out the ones which have already been labeled
       and darken the ones or have some indicator of the superpixels to be labeled - done, sending out a json for the
       superpixel coordinates to be labeled. # TODO: Need to put in a loop for all the images/tiles
    2) Call Sam's API to automatically create a labeling job with a set of s3 or DB image paths and & contours json #TODO
    3) Load these images (created in step 2) in Nick's annotation tool,
    4) Make the superpixels editable to fill it in with color as per the class- TBD by Nick. overlay the superpixel
       coordinates on the image and fill them with colors pertaining to a specific class.
    5) Send these labeled images back to the DB or s3 which contains the rgb images and labels
    """


if __name__ == "__main__":
    superpixels = imread(
        "../../ViewAL_copy/ViewAL/dataset/scannet-sample/raw/selections/superpixel/scene0007_00_000140.png"
    )
    rgb = imread(
        "../../ViewAL_copy/ViewAL/dataset/scannet-sample/raw/selections/color/scene0007_00_000140.jpg"
    )
    label_or_not = imread(
        "../../ViewAL_copy/ViewAL/runs/scannet-sample/regional_viewmckldiv_spx_1_7x2_lr-0.0004_bs-6_ep-60_wb-0_lrs-1_240x320/runs_037/selections/scene0007_00_000140.png"
    )

    generate_json_for_label(superpixels, label_or_not, rgb)
