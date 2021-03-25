import os
import sys
import glob
import numpy as np
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean

'''
animeface is the package which could automatically recognize the face portion of anime images,
'''

import animeface # pip install animeface

def return_images_address(ori_address):
    '''
    ori_add(string) indicates the address of file which hold the original anime image dataset.
    In the original research, this file hold several sub-files which take detail sample images inside.
    
    Return value(list): data is the list of address for the anime images.
    '''
    file_dir = os.path.join(os.getcwd(), ori_address)
    file_list = [f for f in os.listdir(file_dir)]
    file_list_name = [os.path.join(file_dir, f) for f in file_list]
    data = []
    
    for i in range(len(file_list_name)):
        data += glob.glob(os.path.join(file_list_name[i], "*.jpg"))

    return data

def read_image(image_add):
    return_images_address(ori_address):
    '''
    image_add indicates the address of a specify sample anime image.
    
    Return value(PIL.Image): PIL.Image object which includes information from the image in image_add.
    '''
    im = Image.open(image_add)
    return im

def split_fp(im):
    '''
    This function take image array: im(np.array).
    In detail, animeface API was applied to extract the coordinates of the four corners of anime face inside of this image.
    
    Return value(np.array): coordinates of the four corners of anime face inside of this image.
    '''
    faces = animeface.detect(im)
    if faces == []:
        return False
    else:
        fp = faces[0].face.pos
        list_1 = str(fp).split(';')
        list_1_1 = str(list_1[0]).split('(')
        list_1_2 = str(list_1[1]).split(')')
        list_1_1_1 = str(list_1_1[1]).split(',')
        list_1_2_1 = str(list_1_2[0]).split(',')
        list_final = [int(list_1_1_1[1]), int(list_1_1_1[1])+int(list_1_2_1[1]), int(list_1_1_1[0]), int(list_1_1_1[0])+int(list_1_2_1[0])]

        return np.array(list_final)
    
def return_processed_image(img):
    '''
    This function take image object: img(PIL.Image).
    In detail, based on the coordinates of the four corners of anime face inside of this image found from split_fp(),
    anime face of this img could be cropped by these coordinates.
    
    Return value(np.array): image portion which includes the anime face inside of img
    '''
    image = np.array(img)
    #print(img_.max(), img_.min())
    face_size = split_fp(im = img)
    if face_size is False:
        return False
    else:
        d_ = int(face_size[1] - face_size[0])
        d_add = int(d_/2.)

        if face_size[0]-d_add < 0 or face_size[1]+d_add > image.shape[0] or face_size[2]-d_add < 0 or face_size[3]+d_add > image.shape[1]:
            e_1 = face_size[0] - 0
            e_2 = image.shape[0] - face_size[1]
            e_3 = face_size[2] - 0
            e_4 = image.shape[1] - face_size[2]
            e = np.array([e_1, e_2, e_3, e_4]).min()
        else:
            e = d_add

        return image[face_size[0]-e:face_size[1]+e, face_size[2]-e: face_size[3]+e]
    
def resize_processed_image(img, size = 96):
    '''
    This function take image object: img(PIL.Image).
    If these is a anime face inside of this object, the correspond anime face: image_return(np.array) would be return.
    else, False would be returned.
    
    Also, the anime face would be fixed into similar size as (size * size * 3).
    
    Return value: image_return(np.array) if anime face exists in img else False.
    '''
    image = return_processed_image(img = img)
    if image is False:
        return False
    else:
        image_return = resize(image, (size, size, 3), anti_aliasing=True)
        return image_return
    
def data_produce(ori_address):
    '''
    This function take ori_address: address of file which hold the original anime image dataset(as input for return_images_address()).
    Return value: Standardized Concatenate array(np.array) with shape(Num * size * size * 3).
    Num is the # of number of anime face images.
    '''
    images_add = return_images_address(ori_address = ori_address)
    data_final = []
    for i in range(len(images_add)):
        try:
            im = read_image(images_add[i])
        except IOError:
            continue
    im_face_resized = resize_processed_image(img = im, size = img_same)
    if im_face_resized is False:
        continue
    else:
        data_final.append(im_face_resized)
    #print("Now, the raw-data is gonna processing")
    if i % 10 == 0 or i == len(images_add) - 1:
        print_ = "\r- For now, we are finishing {0:.2%} raw-data processing".format(float(i/(len(images_add)-1)))
        sys.stdout.write(print_)
        sys.stdout.flush()

    return np.array(data_final) * 2.0 - 1.0