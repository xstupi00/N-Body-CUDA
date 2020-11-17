import h5py
import numpy as np

filename = "../sampledata/sampleOutput.h5"
filename1 = "step4Output.h5"

h5 = h5py.File(filename, 'r')

vel_x = h5['vel_z']  # VSTOXX futures data

print(vel_x[0])

cnt = 0
for x in vel_x:
    if not np.all(x == x[0]):
        print("ERROR")
    cnt += 1
print(cnt)

h5.close()