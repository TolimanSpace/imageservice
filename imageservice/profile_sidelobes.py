import cProfile
import logging
import glob
import numpy as np
from PIL import Image
from workers.processing import find_centroid, find_stars
from workers.compression import crop_centre, crop_sidelobes, crop_sidelobes_old

logging.basicConfig(level=logging.DEBUG)

# raw_files = glob.glob("images/raw/*.npy")

# raw_image = np.load(raw_files[0])

raw_image = np.asarray(Image.open("toliman_image_0.png"))

centroid_data = find_centroid(raw_image)
core = crop_centre(raw_image, centroid_data["x"], centroid_data["y"])
star_poss = find_stars(core)
x_poss = np.round(star_poss['xs'] + centroid_data['x'] - core.shape[1]//2)
y_poss = np.round(star_poss['ys'] + centroid_data['y'] - core.shape[0]//2)

with cProfile.Profile() as pr:
    sidelobes = crop_sidelobes(raw_image, x_poss, y_poss, centroid_data)

    pr.print_stats(sort='cumtime')