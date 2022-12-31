import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
from skimage import filters

IMG_PX_SIZE = 2048
train = pd.read_csv('data\\train.csv')
train = train[train['cancer']==1]

def histogram_equalization(img):
    m = int(np.max(img))
    hist = np.histogram(img, bins=m+1, range=(0, m+1))[0]
    hist = hist/img.size
    cdf = np.cumsum(hist)
    s_k = (255 * cdf)
    img_new = np.array([s_k[i] for i in img.ravel()]).reshape(img.shape)
    return img_new

def get_masks_and_sizes_of_connected_components(img_mask):
    """
    Finds the connected components from the mask of the image
    """
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels+1):
        this_mask = (mask == i)
        if img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)
    return mask, mask_pixels_dict


def get_mask_of_largest_connected_component(img_mask):
    """
    Finds the largest connected component from the mask of the image
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
    largest_mask = mask == largest_mask_index
    return largest_mask

def image_processing(img):
    threshold = filters.threshold_isodata(img)
    bin_img = (img > threshold)*1
    kernel = np.ones((5, 5), np.uint8)
    bin_img = bin_img.astype('uint8')
    bin_img = cv2.erode(bin_img, kernel, iterations=-2)
    
    img_mask = get_mask_of_largest_connected_component(bin_img)
    
    farthest_pixel = np.max(list(zip(*np.where(img_mask == 1))), axis=0)
    nearest_pixel = np.min(list(zip(*np.where(img_mask == 1))), axis=0)
    cropped =  img[nearest_pixel[0]:farthest_pixel[0], nearest_pixel[1]:farthest_pixel[1]]
    return cropped

def show_multi_img(img_list:list,labels=None,columns=1,rows=1):
    fig = plt.figure(figsize=(12, 12))
    count = 1
    for j in range(len(img_list)):
        for i in range(3):
            img = img_list[j][i]
            label = labels[i]
            fig.add_subplot(rows, columns, count)
            plt.title(label)
            plt.imshow(img,cmap='gray')
            count += 1
    plt.tight_layout()
    plt.show()

for iD in train['patient_id'].unique()[:1]:
    print(f'Patient_ID: {iD}')
    length = train[train['patient_id']==iD]['image_id'].values.shape[0]
    images=[]
    for num in train[train['patient_id']==iD]['image_id']:
        in_path = f'./images/{iD}/{num}.dcm'
        ds = pydicom.dcmread(in_path) #Read .dcm file
        img = np.array(ds.pixel_array, dtype=np.uint8) # Extract image information
        print(f'\tIMG_ID: {num}\tIMG_SHAPE: {img.shape}')
        new_img = image_processing(img)
        eq_img = histogram_equalization(new_img)
        images.append([img,new_img,eq_img])
        print(f'\t\t\t\tIMG_SHAPE: {new_img.shape}')
    labels=['Before Procescing','After Processcing','After CLAHE']
    show_multi_img(images,labels,3,length)

