from Data.data_all import data_intoh5, read_data
from Data.data_tags import data_produce_tags, read_data_tags
from image_PLot.image_plot import plot_images
from Model.DCGAN import DCGAN
from Model.Transfer_Learning import Transfer
import numpy as np
import h5py
'''
The portion is used to training DCGAN for genrating general anime faces in all tags.
'''
#ori_address
#ori_address_2
#data_store_add

data_intoh5(ori_address, ori_address_2, data_store_add)
data = read_data(data_store_add)

plot_images(data)

#model_store_path
#image_store_path
#img_noise
#num_channels
#img_size

model = DCGAN(model_store_path, image_store_path, img_noise, num_channels, img_size)
model.train(epochs, data)

'''
The portion is used to training DCGAN for genrating general anime faces in a specially tag based on model from process above.
'''
#tag_name
#tag_file_address
#data_store_add2

#data_produce_tags(tag_name, tag_file_address, ori_add, data_store_add2)
#tag_data = read_data_tags(tag_name, data_store_add2)

#plot_images(tag_data)

#model_path
#model_store_path2
#image_store_path2
#img_noise2
#num_channels2
#img_size2

#model = Transfer(model_path, model_store_path2, image_store_path2, img_noise2, num_channels2, img_size2)
#model.train(epochs, tag_data)