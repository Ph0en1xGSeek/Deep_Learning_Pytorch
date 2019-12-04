import torch
import numpy as np

arr = [1,2,3,4,5,6,7,8]

brr = np.array([1,2,3,4,5,6,7,8])

crr = np.array(tuple(map(lambda s: s > 2, arr)))

drr = np.array([0,0,1])

print(crr, drr)

print(brr[crr])
print(brr[drr])