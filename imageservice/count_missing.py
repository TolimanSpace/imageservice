from datetime import datetime
import os
import ast

log_file = "/home/toliman-dev/toliman/imageservice/imageservice/process_log.txt"

skipped_frames = 0

with open(log_file, "r") as file:
    for line in file:
        if 'WARNING' in line:
            data = line.strip()
            skipped_frames = skipped_frames + int(data.split()[-3])

print(f"Total skipped frames: {skipped_frames}")

skipped_frames = 0
seq_no = -1

image_path = "/home/toliman-dev/toliman/imageservice/imageservice/images/raw/"
image_files = sorted(os.listdir(image_path))

for file in image_files:
    if file.split('.')[1] == 'npy':
        continue
    elif file.split('.')[1] == 'txt':
        with open(os.path.join(image_path,file), "r") as f:
            for line in f:
                data = ast.literal_eval(line.strip())
                if (data['SEQNUM']) - seq_no > 1:
                    skipped_frames = skipped_frames + data['SEQNUM'] - seq_no - 1
                seq_no = data['SEQNUM']

print(f"Total skipped frames: {skipped_frames}")
