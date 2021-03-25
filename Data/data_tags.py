import os
import sys
import glob
import h5py
import numpy as np
import pandas as pd
from PIL import Image

def return_tags(tags_add):
    '''
    tags_add(string) is the address of csv file for containing the names of tags for different anime faces.
    This function shows all the tags for a certain dataset.
    
    Return value: tags(list) which includes all tag names.
    '''
    head = pd.read_csv(tags_add)
    tags = list(set(head['tag'])) 
    print("The number of species of tags: ", len(tags))
    print("Index:\ttag_name:")
    for i in range(len(tags)):
        print("{0} \t{1}".format(i, tags[i]))
    print("Please choose and copy what tag you want in the next input.")
    return tags

def data_intoh5_tags(data, data_store_add):
    '''
    data(np.array) is the array which contain all the anime faces from a certain tag.
    data_store_add(string) is the address of file(.h5) which hold data.
    This function store data to address as data_store_add.
    '''
    print("\n- The dataset is storing into target address...")
    save_add = os.path.join(os.getcwd(), data_store_add)
    with h5py.File(save_add, 'w') as hy:
        hy.create_dataset('data', data = data)
    print("- Congratulations, the dataset have been successfully stored!!!")

def data_produce_tags(tag_name, tag_csv_add, ori_add, data_store_add):
    '''
    tag_name(string) is the target tag you want to extract anime faces from the dataset
    tag_csv_add(string) is the address of the csv file including the target names.
    ori_add(string) is the address of file which hold the original anime image dataset.
    In the original research, this file hold several sub-files which take detail sample images inside.
    data_store_add(string) is the address of .h5 file you want to store the array of anime faces in.
    
    This function is used to extract anime faces in tag: tag_name from ori_add and store the concatenated dataset in data_store_add.
    '''
    return_tags(tag_csv_add)
    data = []
    data_add = os.path.join(ori_add, tag_name)
    data += glob.glob(os.path.join(data_add, "*.jpg"))
    data += glob.glob(os.path.join(data_add, "*.png"))

    print("Now, the dataset with tag as {0} are being processed".format(tag_name))
    data_final = []
    for i in range(len(data)):
    try:
        im = read_image(data[i])
    except IOError:
        continue
    im_ = np.array(im)
    data_final.append(im_)
    #print("Now, the prepared-data is gonna processing")
    if i % 10 == 0 or i == len(data) - 1:
        print_ = "\r- For now, we are finishing {0:.2%} prepared-data processing with tag as: {1}".format(float(i/(len(data)-1)), tag_name)
        sys.stdout.write(print_)
        sys.stdout.flush()

    data_final = np.array(data_final, dtype = np.float32)/127.5 - 1.0
    tags_store_add = os.path.join(data_store_add, "data_{0}.h5".format(tag_name))
    data_intoh5_tags(data = data_final, data_store_add = tags_store_add)

def read_data_tags(tag_name, data_store_add):
     '''
    tag_name(string) is the target tag you want to extract anime faces from the dataset
    data_store_add(string) is the address of .h5 file you stored the array of anime faces in(should be similar as data_store_add in data_produce_tags()).
    
    This function is used to read anime faces array in tag:tag_name from address as data_store_add.
    '''
    tags_store_add = os.path.join(data_store_add, "data_{0}.h5".format(tag_name))
    with h5py.File(tags_store_add, 'r') as hy:
        data = np.array(hy.get('data'))

    return data