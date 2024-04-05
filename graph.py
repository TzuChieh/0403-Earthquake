import csv
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def magnitude_to_energy(magnitude):
    return 10.0 ** (4.8 + 1.5 * magnitude)


def energy_to_magnitude(energy):
    return (np.log10(energy) - 4.8) / 1.5


class EarthquakeData:
    def __init__(self):
        self.names = []
        self.times = []
        self.longitudes = []
        self.latitudes = []
        self.magnitudes = []
        self.depths = []
        self.location_descriptions = []

    def process(self):
        # Calculate running average of earthquake counts
        avg_hours = 4.0
        half_hours = timedelta(hours=avg_hours / 2)
        self.counts = []
        for time in self.times:
            self.counts.append(self.count(time - half_hours, time + half_hours) / avg_hours)

        # Calculate cumulative energy release amount
        total_energy = 0.0
        self.cumulative_magnitudes = []
        for magnitude in reversed(self.magnitudes):
            total_energy += magnitude_to_energy(magnitude)
            self.cumulative_magnitudes.insert(0, energy_to_magnitude(total_energy))

    def count(self, begin_time, end_time):
        count = 0
        for time in self.times:
            if time >= begin_time and time < end_time:
                count += 1
        return count


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
            data.magnitudes.append(float(row[4]))
            data.depths.append(float(row[5]))
            data.location_descriptions.append(row[6])

    data.longitudes = np.array(data.longitudes)
    data.latitudes = np.array(data.latitudes)
    data.magnitudes = np.array(data.magnitudes)
    data.depths = np.array(data.depths)

    return data


# Use a non-interactive backend so plot window will not pop out
matplotlib.use('Agg')

# Creates output directory (this won't be tracked)
Path("./outputs/").mkdir(parents=True, exist_ok=True)

data = read_data("data/20240405_0742.csv", "2024-04-03 07:58:00", "2025-04-03 07:58:00")
data.process()

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()
ax.bar(
    data.times, 
    data.magnitudes,
    width=0.0075,
    alpha=0.5,
    color=np.where(data.magnitudes >= 5.5, 'r', np.where(data.magnitudes < 4, 'b', 'g')))
ax.set_xlabel("Time")
ax.set_ylabel("Magnitude")
plt.savefig(Path("./outputs/mag_t.png"))
plt.clf()

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()
ax.plot(
    data.times, 
    data.counts)
ax.set_xlabel("Time")
ax.set_ylabel("Counts (4 hours mean)")
plt.savefig(Path("./outputs/count_t.png"))
plt.clf()

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()
ax.plot(
    data.times, 
    data.cumulative_magnitudes)
ax.set_xlabel("Time")
ax.set_ylabel("Cumulative Energy")
plt.savefig(Path("./outputs/energy_t.png"))
plt.clf()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(
    data.longitudes, 
    data.latitudes,
    data.depths,
    c=data.depths,
    s=data.magnitudes ** 3.8,
    alpha=0.6,
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
