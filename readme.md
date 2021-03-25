# Anime Faces generating by DCGAN

This is a project about applying Deep Convolutional Generative Adversarial Networks(DCGAN) in Keras for anime faces generating.

# Dataset

The original dataset applied for training of DCGAN could be divided into two parts:
1. The raw-data:
For this part, the anime faces is cropped and collected from public dataset with various anime images. The API used in cropping
anime faces is animeface which is good packages for detecting faces in anime images. The detail code for cropping anime faces 
based on detected result from animeface in certain images could be found in 'data_original.py' in 'Data' folder.

However, in practical using, since animeface could only detect faces from few obvious anime images from overall dataset, collected 
anime faces are not enough for building training data. Further prepared data should be required.

2. Prepared data:
In this project, most anime face cases is pick from sharing link from https://drive.google.com/file/d/0B4wZXrs0DHMHMEl1ODVpMjRTWEk/view?usp=sharing.
over 100000 anime faces with similar size as 96 * 96 * 3 are included, which is enough for training.

# Structure of project
1. Data Processing:
In this project, images of anime face was concatenated in array with range as [-1,1] and stored in .h5 format for fast access.
In detail:
Data/data_original.py was implemented for extracting anime faces from raw-data.
Data/data_prepared.py was implemented for extracting anime faces from prepared data.
Data/data_all.py was implemented for combining arrays from both above and storing in .h5 file. Besides, further reading function was also attached inside.
Data/data_tags.py was designed for extracting anime faces from prepared data in specific tag. Also, storing and reading method in .h5 file were attached.

2. Models:
In this project, DCGAN was applied for two steps under considering generating general anime faces and faces in specific tag:
In detail:
Models/DCGAN.py： Inspired by link: https://github.com/eriklindernoren/Keras-GAN, generator and discrimator was built in stacks where each is built by
Conv2DTranspose -> BatchNormalization -> Relu -> UpSampling2D. Besides, embed ploting and restoring function is included for recording trained models and 
correspond generated faces for every 100 epochs.
Models/Transfer_Learning.py： Based on the trained model above, transfer learning model is developed for further generateing anime faces in specific tags. 
The basic idea is that, appending new stacks or layers at the end of generator and training the updated model with dataset in correspond tag only. Since 
the time limit, this part still not be tested.


3. Others:
Image_Plot: This file contain the plot function for estalishing 100 anime faces randomly from target dataset.
test.py: The main file for do sample run of the DCGAN models.



# Resource
https://github.com/eriklindernoren/Keras-GAN
https://drive.google.com/file/d/0B4wZXrs0DHMHMEl1ODVpMjRTWEk/view?usp=sharing
https://github.com/nagadomi/lbpcascade_animeface