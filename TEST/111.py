import numpy as np

a=np.array([1,2,4])
b=np.array([1,2,4,11])
c=np.concatenate((a,b),axis=0)
print(c)