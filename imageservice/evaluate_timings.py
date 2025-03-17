import ast
import os

from astropy.io import fits
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


image_path = "/home/toliman-dev/toliman/imageservice/imageservice/images/compressed/"
image_files = os.listdir(image_path)

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

comtimes = []
camtimes = []

for file in image_files:
    if file.split('.')[1] == 'fits':
        with fits.open(os.path.join(image_path, file)) as hdul:
            comtime = datetime.strptime(hdul[0].header['COMTIME'], '%Y-%m-%d %H:%M:%S.%f')
            camtime = int(hdul[0].header['CAMTIME'])

            comtimes.append(comtime)
            camtimes.append(camtime)
    elif file.split('.')[1] == 'png':
        camtime = int(file.split('.')[0].split('_')[1])

        camtimes.append(camtime)


fig, ax = plt.subplots(nrows = 2, figsize = (10,10))

plt.sca(ax[0])
plt.plot(np.sort(camtimes)[1:]/1e9,np.diff(np.sort(camtimes))/1e9,'.')
plt.ylabel('$\Delta$t (seconds)')
plt.axhline(0.1, c='r', alpha=0.5)
plt.ylim(0,2)
plt.title('Camera Timestamp')

plt.sca(ax[1])
plt.plot(np.sort(camtimes)[1:]/1e9,np.diff(np.sort(camtimes))/1e3 - 1e5,'.')
plt.xlabel('Camera Time (seconds)')
plt.ylabel('$\Delta$t - $10^5$ ($\mu$s)')
plt.axhline(0, c='r')
plt.ylim(-10,10)

plt.savefig(f"benchmarking/camera_timing_{timestamp}.pdf", facecolor='w')

if len(comtimes) > 0:

    fig, ax = plt.subplots(nrows = 1, figsize = (10,5))

    plt.plot(np.sort(comtimes)[1:],[dt.total_seconds() for dt in np.diff(np.sort(comtimes))],'.')
    plt.xlabel('Computer Time (H:M:S)')
    plt.ylabel('$\Delta$t (seconds)')
    plt.ylim(0,2)
    plt.axhline(0.1,c='r')
    plt.title('Computer Timestamp')

    plt.savefig(f"benchmarking/computer_timing_{timestamp}.pdf", facecolor='w')

# Centroids

centroid_file = "/home/toliman-dev/toliman/imageservice/imageservice/centroids.txt"

xs, ys, ts = [], [], []

with open(centroid_file, "r") as file:
    for line in file:
        data = ast.literal_eval(line.strip())
        xs.append(data["x"])
        ys.append(data["y"])
        ts.append(datetime.strptime(data["t"], '%Y-%m-%d %H:%M:%S.%f'))

if len(ts) > 0:
    fig, ax = plt.subplots(nrows = 1, figsize = (10,5))

    plt.plot(ts[1:],[dt.total_seconds() for dt in np.diff(ts)],'.')
    plt.xlabel('Centroid Time (seconds)')
    plt.ylabel('$\Delta$t (seconds)')
    plt.axhline(0.1,c='r')
    plt.ylim(0,4)
    plt.title('Centroid Timestamp')

    plt.savefig(f"benchmarking/centroid_timing_{timestamp}.pdf", facecolor='w')