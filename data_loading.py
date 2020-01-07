import numpy as np

data_store = []
filename = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\test_data.txt'
with open(filename, 'r') as f:
    x = f.readlines()
    ooo = x[0].replace('\n',' ')
    ooo = ooo.split(' ')
    ooo = ooo[:-1]
    ooo = np.array(ooo)
    ooo = ooo.astype(np.float)
    data_store.append(ooo)