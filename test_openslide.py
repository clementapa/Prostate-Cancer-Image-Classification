import os
import random
import openslide
import numpy as np

path_im = os.path.join(
    os.getcwd(), "assets/mvadlmi/train/train/0a619ab32b0cd639d989cce1e1e17da0.tiff"
)

wsi = openslide.OpenSlide(path_im)


def return_random_patch(whole_slide):
    wsi_dimensions = whole_slide.dimensions
    random_location_x = random.randint(0, wsi_dimensions[0] - 256)
    random_location_y = random.randint(0, wsi_dimensions[1] - 256)
    cropped_image = whole_slide.read_region(
        (random_location_x, random_location_y), 0, (256, 256)
    )
    while (
        np.sum(
            np.any(np.array(cropped_image)[:, :, :-1] == [255.0, 255.0, 255.0], axis=-1)
        )
        > 0.2 * 256 * 256
    ):
        random_location_x = random.randint(0, wsi_dimensions[0] - 256)
        random_location_y = random.randint(0, wsi_dimensions[1] - 256)
        cropped_image = whole_slide.read_region(
            (random_location_x, random_location_y), 0, (256, 256)
        )
    return cropped_image


im = return_random_patch(wsi)

print("end")
