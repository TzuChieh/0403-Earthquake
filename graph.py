import csv
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


time_format = '%Y-%m-%d %H:%M:%S'
mainshock_time = datetime.strptime("2024-04-03 07:58:09", time_format)

# Use a non-interactive backend so plot window will not pop out
matplotlib.use('Agg')


def magnitude_to_energy(magnitude):
    return 10.0 ** (4.8 + 1.5 * magnitude)

def energy_to_magnitude(energy):
    # To prevent log(0)
    epsilon = 1e-14
    return (np.log10(energy + epsilon) - 4.8) / 1.5

def make_time_range(begin_time, end_time, step_time):
    times = []
    while begin_time < end_time:
        times.append(begin_time)
        begin_time += step_time
    return times


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
        """
        Process all data and store additional information. This method should be called only after all modifications
        are done for the data.
        """
        self.longitudes = np.array(self.longitudes)
        self.latitudes = np.array(self.latitudes)
        self.magnitudes = np.array(self.magnitudes)
        self.depths = np.array(self.depths)

        # Calculate running average of earthquake counts
        one_hour = timedelta(hours=1)
        avg_hours = 4.0
        half_avg_hours = timedelta(hours=avg_hours / 2)
        self.time_and_counts = []
        for time in make_time_range(self.times[-1], self.times[0] + one_hour, one_hour):
            avg_count = self.count(time - half_avg_hours, time + half_avg_hours) / avg_hours
            self.time_and_counts.append((time, avg_count))

        # Calculate energy release amount in time slot
        self.time_and_summed_magnitudes = []
        for time in make_time_range(self.times[-1], self.times[0] + one_hour, one_hour):
            e = self.energy(time - half_avg_hours, time + half_avg_hours)
            m = energy_to_magnitude(e)
            if m > 0.0:
                self.time_and_summed_magnitudes.append((time, m))

        # Calculate cumulative energy release amount
        total_e = 0.0
        self.cumulative_magnitudes = []
        for magnitude in reversed(self.magnitudes):
            total_e += magnitude_to_energy(magnitude)
            m = energy_to_magnitude(total_e)
            self.cumulative_magnitudes.insert(0, m)

    def count(self, begin_time, end_time):
        """
        @return Number of earthquakes in the specified time range.
        """
        count = 0
        for time in self.times:
            if time >= begin_time and time < end_time:
                count += 1
        return count
    
    def energy(self, begin_time, end_time):
        """
        @return Total energy in the specified time range.
        """
        e = 0.0
        for i, time in enumerate(self.times):
            if time >= begin_time and time < end_time:
                e += magnitude_to_energy(self.magnitudes[i])
        return e
    
    def remove_entries_by(self, longitude_range=(0.0, 0.0), latitude_range=(0.0, 0.0)):
        for i in reversed(range(0, len(self.names))):
            long, lat = (self.longitudes[i], self.latitudes[i])
            if (longitude_range[0] <= long and long < longitude_range[1] and
                latitude_range[0] <= lat and lat < latitude_range[1]):
                self.remove_entry(i)
    
    def add_entry(self, name, time, longitude, latitude, magnitude, depth, location_description):
        self.names.append(name)
        self.times.append(time)
        self.longitudes.append(longitude)
        self.latitudes.append(latitude)
        self.magnitudes.append(magnitude)
        self.depths.append(depth)
        self.location_descriptions.append(location_description)

    def remove_entry(self, idx):
        del self.names[idx]
        del self.times[idx]
        del self.longitudes[idx]
        del self.latitudes[idx]
        del self.magnitudes[idx]
        del self.depths[idx]
        del self.location_descriptions[idx]


def read_data(file_name, begin_time, end_time):
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

            data.add_entry(
                name=row[0],
                time=time,
                longitude=float(row[2]),
                latitude=float(row[3]),
                magnitude=float(row[4]),
                depth=float(row[5]),
                location_description=row[6])
            
    return data


# Creates output directory (this won't be tracked)
Path("./outputs/").mkdir(parents=True, exist_ok=True)

data = read_data("data/20240427_0320.csv", "2024-04-03 07:58:00", "2025-04-03 07:58:00")

# Exclude earthquakes on the west side of Taiwan
data.remove_entries_by(longitude_range=(0.0, 120.45), latitude_range=(0.0, 90.0))

data.process()

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()
ax.bar(
    data.times, 
    data.magnitudes,
    width=0.01 * data.magnitudes,
    alpha=0.5,
    color=np.where(data.magnitudes >= 5, 'r', np.where(data.magnitudes < 4, 'b', 'g')))
ax.set_xlabel("Time")
ax.set_ylabel("Magnitude")
ax.grid()
plt.savefig(Path("./outputs/mag_t.png"))
plt.clf()

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()
ax.plot(
    [tc[0] for tc in data.time_and_counts], 
    [tc[1] for tc in data.time_and_counts])
ax.set_xlabel("Time")
ax.set_ylabel("Counts (4 hours mean)")
ax.grid()
plt.savefig(Path("./outputs/count_t.png"))
plt.clf()

hours_after_mainshock = [(t - mainshock_time).total_seconds() / 3600.0 for t in [tc[0] for tc in data.time_and_counts]]
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()
ax.plot(
    hours_after_mainshock, 
    [tc[1] for tc in data.time_and_counts])
ax.set_xlabel("Hours after mainshock (log scale)")
ax.set_ylabel("Counts (4 hours mean, log scale)")
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
ax.set_xlim(2.0)
ax.grid()
plt.savefig(Path("./outputs/count_t_flog.png"))
plt.clf()

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()
ax.plot(
    [tc[0] for tc in data.time_and_summed_magnitudes],
    [tc[1] for tc in data.time_and_summed_magnitudes])
ax.set_xlabel("Time")
ax.set_ylabel("Energy Release")
ax.grid()
plt.savefig(Path("./outputs/summed_energy_t.png"))
plt.clf()

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot()
ax.plot(
    data.times, 
    data.cumulative_magnitudes)
ax.set_xlabel("Time")
ax.set_ylabel("Cumulative Energy")
ax.grid()
plt.savefig(Path("./outputs/cumulative_energy_t.png"))
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
