import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import OrdinalEncoder
import pydicom
from imageio.v3 import imwrite

train = pd.read_csv('data\\train.csv')
for iD in train['patient_id'].unique():
    for num in train[train['patient_id']==iD]['image_id']:
        try:
            in_path = f'./images/{iD}/{num}.dcm'
            ds = pydicom.read_file(in_path) #Read .dcm file
            img = ds.pixel_array # Extract image information
            print(img.shape)
            plt.imshow(img)
            plt.show()
        except:
            pass
    plt.close()