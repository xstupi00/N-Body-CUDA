import h5py
import numpy as np

filename = "../sampledata/sampleOutput.h5"
filename1 = "step0Output.h5"

h5 = h5py.File(filename, 'r')

pos_x = h5['pos_x']  # VSTOXX futures data

print(pos_x[0][1])

# cnt = 0
# for x in vel_x:
#     if not np.all(x == x[0]):
#         print("ERROR")
#     cnt += 1
# print(cnt)

h5.close()