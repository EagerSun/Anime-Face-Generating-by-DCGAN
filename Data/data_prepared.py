import os
import sys
import glob
import numpy as np
from PIL import Image

def return_images_address_pre(ori_address):
    '''
    ori_address(string) indicates the address of file which hold the original anime image dataset.
    In the original research, this file hold several sub-files which take detail sample images inside.
    
    Return value(list): data is the list of address for the anime images.
    '''
    file_dir = os.path.join(os.getcwd(), ori_address)
    file_list = [f for f in os.listdir(file_dir)]
    file_list_name = [os.path.join(file_dir, f) for f in file_list]
    data = []
    
    for i in range(len(file_list_name)):
        data += glob.glob(os.path.join(file_list_name[i], "*.jpg"))
        data += glob.glob(os.path.join(file_list_name[i], "*.png"))
        
    return data

def data_produce_pred(ori_address):
    '''
    This function take ori_address: address of file which hold the original anime image dataset(as input for return_images_address_pre()).
    Return value: Standardized Concatenate array(np.array) with shape(Num * size * size * 3).
    Num is the # of number of anime face images.
    '''
    data_add = return_images_address_pre(ori_address = ori_address)
    data_final = []
    for i in range(len(data_add)):
        try:
            im = read_image(data_add[i])
        except IOError:
            continue
    image = np.array(im)
    data_final.append(image)
    #print("Now, the prepared-data is gonna processing")
    if i % 1000 == 0 or i == len(data_add) - 1:
        string = "\r- For now, we are finishing {0:.2%} prepared-data processing".format(float(i/(len(data_add)-1)))
        sys.stdout.write(string)
        sys.stdout.flush()

    data_final = np.array(data_final, dtype = np.float32)
    return data_final/127.5 - 1.0

