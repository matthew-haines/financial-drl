import pandas as pd 
import numpy as np
from math import floor
from datetime import datetime

# Data is in minutes
starting_min = 1000000
length = 15
raw_data = pd.read_csv('data/bitstamp_minute_data.csv', usecols=[0, 1, 2, 3, 4, 5], skiprows=starting_min)
array = raw_data.values

out_length = floor(array.shape[0] / length)
out_array = np.zeros((out_length, 8))
for i in range(out_length-16):
    out_array[i, 0] = np.sin(2 * np.pi * (float(datetime.fromtimestamp(array[i+15-1, 0] / 1000).strftime('%-M')) / 60.0)) # Minutes
    out_array[i, 1] = np.sin(2 * np.pi * (float(datetime.fromtimestamp(array[i+15-1, 0] / 1000).strftime('%-H')) / 24)) # Hours
    out_array[i, 2] = np.sin(2 * np.pi * (float(datetime.fromtimestamp(array[i+15-1, 0] / 1000).strftime('%w')) / 7)) # Weekday

    out_array[i, 3] = array[i * length, 0] # Open
    out_array[i, 4] = np.max(array[i*length:(i+1)*length, 1]) # High
    out_array[i, 5] = np.max(array[i*length:(i+1)*length, 2]) # Low
    out_array[i, 6] = array[i * length + (length - 1), 3] # Close
    out_array[i, 7] = np.sum(array[i*length:i*(length+1), 4]) # Volume

np.savetxt('data/prepared_15.txt', out_array)
