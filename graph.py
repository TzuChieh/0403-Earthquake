import csv
from pathlib import Path
from datetime import datetime
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class EarthquakeData:
    def __init__(self):
        self.names = []
        self.times = []
        self.longitudes = []
        self.latitudes = []
        self.magitudes = []
        self.depths = []
        self.location_descriptions = []

def read_data(file_name, begin_time, end_time):
    time_format = '%Y-%m-%d %H:%M:%S'
    begin_time = datetime.strptime(begin_time, time_format)
    end_time = datetime.strptime(end_time, time_format)
    data = EarthquakeData()
    with open(Path(__file__).parent / file_name, newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)

        # First two lines are header, we do not need them
        next(csv_reader)
        next(csv_reader)

        for row in csv_reader:
            # Skip entries based on time
            time = datetime.strptime(row[1], time_format)
            if time < begin_time or time > end_time:
                continue

            data.names.append(row[0])
            data.times.append(time)
            data.longitudes.append(float(row[2]))
            data.latitudes.append(float(row[3]))
            data.magitudes.append(float(row[4]))
            data.depths.append(float(row[5]))
            data.location_descriptions.append(row[6])

    data.longitudes = np.array(data.longitudes)
    data.latitudes = np.array(data.latitudes)
    data.magitudes = np.array(data.magitudes)
    data.depths = np.array(data.depths)

    return data


# Use a non-interactive backend so plot window will not pop out
matplotlib.use('Agg')

# Creates output directory (this won't be tracked)
Path("./outputs/").mkdir(parents=True, exist_ok=True)

data = read_data("data/20240405_0742.csv", "2024-04-03 07:58:00", "2025-04-03 07:58:00")

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()
ax.bar(
    data.times, 
    data.magitudes,
    width=0.0075,
    alpha=0.5)
ax.set_xlabel("Time")
ax.set_ylabel("Magnitude")
plt.savefig(Path("./outputs/mag_t.png"))
plt.clf()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(
    data.longitudes, 
    data.latitudes,
    data.depths,
    c=data.depths,
    s=data.magitudes ** 3.5,
    alpha=0.7,
    edgecolors='gray',
    cmap='viridis_r')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Depth (km)")
ax.invert_zaxis()
animated_imgs = []
for deg in range(0, 360, 4):
    ax.view_init(elev=20, azim=deg)
    img_path = f"./outputs/3d_scatter_{deg}.png"
    plt.savefig(img_path)
    animated_imgs.append(Image.open(img_path))
plt.clf()

# Save animated images as .gif
animated_imgs[0].save(
    "./outputs/3d_scatter.gif", 
    save_all=True, 
    append_images=animated_imgs[1:], 
    duration=100, 
    loop=0,
    allow_mixed=True)
