import pandas as pd
import matplotlib.pyplot as plt
import pydicom

train = pd.read_csv('data\\train.csv')
for iD in train['patient_id'].unique()[:5]:
    print(f'Patient_ID: {iD}')
    len = train[train['patient_id']==iD]['image_id'].values.shape[0]
    f,ax = plt.subplots(1,len)
    f.suptitle(f'Patient {iD}', y=0.8)
    i=0
    for num in train[train['patient_id']==iD]['image_id']:
        in_path = f'./images/{iD}/{num}.dcm'
        ds = pydicom.read_file(in_path) #Read .dcm file
        img = ds.pixel_array # Extract image information
        print(f'\tIMG_ID: {num}\tIMG_SHAPE: {img.shape}')
        ax[i].imshow(img)
        ax[i].set_title(f'{num}',fontsize=10)
        i+=1
    f.tight_layout()
    plt.show()