from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection

log_file = "/home/toliman-dev/toliman/imageservice/imageservice/process_log.txt"

versiontime = datetime.now().strftime('%Y%m%d-%H%M%S')

timestamp, process, cpu, state = [], [], [], []

with open(log_file, "r") as file:
    for line in file:
        if 'DEBUG' in line:
            if "all workers finished" in line:
                break
            data = line.strip()
            timestamp.append(datetime.strptime(data[:23], '%Y-%m-%d %H:%M:%S,%f'))
            cpu.append(data.split('cpu-')[1].split(':')[0])
            message = data.split(': ')[-1].split(' ')
            process.append(message[1])

            if 'start' in message:
                state.append(1)
            elif 'end' in message or 'stopping' in message:
                state.append(0)
            else:
                state.append(np.nan)
                
d = {'timestamp': timestamp, 'process': process, 'cpu': cpu, 'state': state}
df = pd.DataFrame(data = d)

timeline_data_df = pd.DataFrame()

for process in df.process.unique():

    df_1 = df.query(f"process=='{process}' and state==1").reset_index()
    df_2 = df.query(f"process=='{process}' and state==0").reset_index()

    process_df = df_1.join(df_2.timestamp, lsuffix = '_start', rsuffix = '_end').drop(columns = ['index', 'state'])

    timeline_data_df = pd.concat([timeline_data_df, process_df])

colourmapping_process = {
    "acquire_frames_setup": "C0",
    "acquire_frames": "C1",
    "frame_distributor": "C2",
    "process_frames": "C3",
    "save_to_disk": "C4",
    "serial_comm": "C5",
    "save_centroid": "C6",
    "actuate_piezo": "C7",
    "compress": "C8"
}

colourmapping_cpu = {
    "0": "C0",
    "1": "C1",
    "2": "C2",
    "3": "C3",
    "4": "C4",
    "5": "C5",
}

cats = {
    "acquire_frames_setup": 1,
    "acquire_frames": 2,
    "frame_distributor": 3,
    "process_frames": 4,
    "save_to_disk": 5,
    "serial_comm": 6,
    "save_centroid": 7,
    "actuate_piezo": 8,
    "compress": 9    
}

verts = []
colours = []
for idx, row in timeline_data_df.iterrows():
    v = [(mdates.date2num(row.timestamp_start), float(row.cpu)-0.4),
         (mdates.date2num(row.timestamp_start), float(row.cpu)+0.4),
         (mdates.date2num(row.timestamp_end), float(row.cpu)+0.4),
         (mdates.date2num(row.timestamp_end), float(row.cpu)-0.4),
         (mdates.date2num(row.timestamp_start), float(row.cpu)-0.4)]
    verts.append(v)
    colours.append(colourmapping_process[row.process])

bars = PolyCollection(verts, facecolors = colours)

fig, ax = plt.subplots(figsize=(15,5))
ax.add_collection(bars)
ax.autoscale()
loc = mdates.MinuteLocator()
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

ax.set_yticks([0,1,2,3,4,5])
ax.set_yticklabels(['CPU 1','CPU 2','CPU 3','CPU 4','CPU 5','CPU 6'])

# legend
labels = list(colourmapping_process.keys())
handles = [plt.Rectangle((0,0),1,1, color=colourmapping_process[label]) for label in labels]
plt.legend(handles,labels, loc = 'lower right')

plt.savefig(f"benchmarking/cpu_usage_{versiontime}.pdf", facecolor='w')


verts = []
colours = []
for idx, row in timeline_data_df.iterrows():
    v = [(mdates.date2num(row.timestamp_start), cats[row.process]-0.4),
         (mdates.date2num(row.timestamp_start), cats[row.process]+0.4),
         (mdates.date2num(row.timestamp_end), cats[row.process]+0.4),
         (mdates.date2num(row.timestamp_end), cats[row.process]-0.4),
         (mdates.date2num(row.timestamp_start), cats[row.process]-0.4)]
    verts.append(v)
    colours.append(colourmapping_cpu[row.cpu])

bars = PolyCollection(verts, facecolors = colours, edgecolors = 'k')

fig, ax = plt.subplots(figsize=(15,5))
ax.add_collection(bars)
ax.autoscale()
loc = mdates.SecondLocator()
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
ax.set_xlim(mdates.date2num(timeline_data_df.timestamp_start.min())-1e-6, mdates.date2num(timeline_data_df.timestamp_start.min()) + 5e-5)
ax.set_yticks([1,2,3,4,5,6,7,8,9])
ax.set_yticklabels(cats.keys())

# legend
labels = list(colourmapping_cpu.keys())
handles = [plt.Rectangle((0,0),1,1, color=colourmapping_cpu[label]) for label in labels]
labels = ['CPU ' + label for label in labels]
plt.legend(handles, labels, loc = 'upper left')

plt.savefig(f"benchmarking/process_timing_start_{versiontime}.pdf", facecolor='w')


verts = []
colours = []
for idx, row in timeline_data_df.iterrows():
    v = [(mdates.date2num(row.timestamp_start), cats[row.process]-0.4),
         (mdates.date2num(row.timestamp_start), cats[row.process]+0.4),
         (mdates.date2num(row.timestamp_end), cats[row.process]+0.4),
         (mdates.date2num(row.timestamp_end), cats[row.process]-0.4),
         (mdates.date2num(row.timestamp_start), cats[row.process]-0.4)]
    verts.append(v)
    colours.append(colourmapping_cpu[row.cpu])

bars = PolyCollection(verts, facecolors = colours)

fig, ax = plt.subplots(figsize=(15,5))
ax.add_collection(bars)
ax.autoscale()
loc = mdates.MinuteLocator()
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
ax.set_yticks([1,2,3,4,5,6,7,8,9])
ax.set_yticklabels(cats.keys())

# legend
labels = list(colourmapping_cpu.keys())
handles = [plt.Rectangle((0,0),1,1, color=colourmapping_cpu[label]) for label in labels]
labels = ['CPU ' + label for label in labels]
plt.legend(handles, labels, loc = 'lower right')

plt.savefig(f"benchmarking/process_timing_{versiontime}.pdf", facecolor='w')


verts = []
colours = []
for idx, row in timeline_data_df.iterrows():
    v = [(mdates.date2num(row.timestamp_start), cats[row.process]-0.4),
         (mdates.date2num(row.timestamp_start), cats[row.process]+0.4),
         (mdates.date2num(row.timestamp_end), cats[row.process]+0.4),
         (mdates.date2num(row.timestamp_end), cats[row.process]-0.4),
         (mdates.date2num(row.timestamp_start), cats[row.process]-0.4)]
    verts.append(v)
    colours.append(colourmapping_cpu[row.cpu])

bars = PolyCollection(verts, facecolors = colours, edgecolors = 'k')

fig, ax = plt.subplots(figsize=(15,5))
ax.add_collection(bars)
ax.autoscale()
loc = mdates.MinuteLocator()
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
ax.set_xlim(mdates.date2num(timeline_data_df.timestamp_start.min())-1e-6, mdates.date2num(timeline_data_df.timestamp_start.min()) + 7.5e-4)
ax.set_yticks([1,2,3,4,5,6,7,8,9])
ax.set_yticklabels(cats.keys())

# legend
labels = list(colourmapping_cpu.keys())
handles = [plt.Rectangle((0,0),1,1, color=colourmapping_cpu[label]) for label in labels]
labels = ['CPU ' + label for label in labels]
plt.legend(handles, labels, loc = 'upper left')

plt.savefig(f"benchmarking/process_timing_firstminute_{versiontime}.pdf", facecolor='w')
