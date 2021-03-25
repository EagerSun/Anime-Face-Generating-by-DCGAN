import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.initializers import RandomNormal

class DCGANS(object):
    def __init__(self, model_store_path, image_store_path, img_noise, num_channels, img_size):
        '''
        model_store_path(string): The address for store the trained models during training.
        image_store_path(string): The address for store the performance/generated anime faces during training.
        img_noise(int): The size of noise images for generating anime faces(img_noise * img_noise).
        num_channels(int): The number of channels for anime faces.
        img_size(int): The size of output anime faces(img_size * img_size * 3).
        
        '''
        self.model_store_path = model_store_path
        self.image_store_path = image_store_path
        self.image_path = image_path
        self.noise_single = img_noise
        self.img_channels = num_channels
        self.noise_dim = self.noise_single*self.noise_single
        self.img_size = img_size
        self.img_flat = self.img_size * self.img_size
        self.img_shape = (self.img_size, self.img_size)
        self.img_shape_full = (self.img_size, self.img_size, self.img_channels)
    
        #define the optimizer of training models.
        optimizer = Adam(lr = 0.0003, beta_1 = 0.5)
        
        #define the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
        #define the generator
        self.generator = self.build_generator()
        noise = Input(shape = (self.noise_dim, ))
        output_image = self.generator(noise)
        
        #FREEZE the discriminator as training discriminator
        self.discriminator.trainable = False

        img_correct = self.discriminator(output_image)
        
        #Build the model and compile it.
        self.combine = Model(noise_, img_correct)
        self.combine.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    def build_generator(self):
        '''
        This function is used to define the generator of model(DCGAN).
        '''
        # layers, kernel_size, strides should be changed wisely to fit the output size.

        model = Sequential()
        model.add(Dense(512*6*6, activation = 'relu', input_shape = (self.noise_dim, )))
        model.add(Reshape((6, 6, 512)))
        model.add(Conv2DTranspose(512, kernel_size = 5, strides = 2, padding = 'same', use_bias = True, kernel_initializer = RandomNormal(stddev = 0.02)))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation("relu"))
        #model.add(UpSampling2D())

        model.add(Conv2DTranspose(256, kernel_size = 5, strides = 2, padding = 'same', use_bias = True, kernel_initializer = RandomNormal(stddev = 0.02)))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation("relu"))
        #model.add(UpSampling2D())

        model.add(Conv2DTranspose(128, kernel_size = 5, strides = 2, padding = 'same', use_bias = True, kernel_initializer = RandomNormal(stddev = 0.02)))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation("relu"))
        #model.add(UpSampling2D())

        model.add(Conv2DTranspose(64, kernel_size = 5, strides = 2, padding = 'same', use_bias = True, kernel_initializer = RandomNormal(stddev = 0.02)))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation("relu"))
        #model.add(UpSampling2D())

        model.add(Conv2DTranspose(self.img_channels, kernel_size = 5, strides = 1,  padding = 'same', use_bias = True, kernel_initializer = RandomNormal(stddev = 0.02)))
        #model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation("tanh"))

        #model.summary()

        noise = Input(shape = (self.noise_dim, ))
        img_output = model(noise)

        return Model(noise, img_output)

    def build_discriminator(self):
        '''
        This function is used to define the discriminator of model(DCGAN).
        '''
        model = Sequential()
        model.add(Reshape((self.img_size, self.img_size, self.img_channels)))
        model.add(Conv2D(32, kernel_size = 5, strides = 2, padding = 'same', use_bias = True, kernel_initializer = RandomNormal(stddev = 0.02)))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(rate = 0.25))
        model.add(Conv2D(64, kernel_size = 5, strides = 2, padding = 'same', use_bias = True, kernel_initializer = RandomNormal(stddev = 0.02)))
        #model.add(ZeroPadding2D(padding = ((0,1),(0,1))))
        model.add(BatchNormalization(momentum = 0.99))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(rate = 0.25))
        model.add(Conv2D(128, kernel_size = 5, strides = 2, padding = 'same', use_bias = True, kernel_initializer = RandomNormal(stddev = 0.02)))
        #model.add(ZeroPadding2D(padding = ((0,1),(0,1))))
        model.add(BatchNormalization(momentum = 0.99))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(rate = 0.25))
        model.add(Conv2D(128, kernel_size = 5, strides = 2, padding = 'same', use_bias = True, kernel_initializer = RandomNormal(stddev = 0.02)))
        
        model.add(BatchNormalization(momentum = 0.99))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(rate = 0.25))
        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))

        img = Input(shape = (self.img_size, self.img_size, self.img_channels))
        output_ = model(img)

        return Model(img, output_)
    
    def noise_data(self, data_read_, mean, std):
        '''
        data_read_(np.array): This is the array with anime faces for training.
        mean(float): mean for Gaussian noise.
        std(float): std for Gaussian noise.
        This function is intend to add slight Gaussian noise in data_read_.
        
        Return value: the noised data_read_
        '''
        noise_ = np.random.normal(loc = mean, scale = std, size = (data_read_.shape[0], data_read_.shape[1], data_read_.shape[2], data_read_.shape[3]))

        return np.clip(data_read_ + noise_, -1, 1).astype(np.float32)

    def shuffle_data(self, train):
        '''
        train(np.array): This is the array with anime faces for training.
        This function is intend to shuffle train in its first axis.
        
        Return value: shuffled train.
        '''
        idx = np.arange(0, len(train))
        np.random.shuffle(idx)
        #print(idx.shape)
        return train[idx]

    def save_and_plot_images(self, iteration, tags, epochs = 100):
        '''
        iteration(int): The index of epoch during training.
        tags(string): The target tag if it exists in this DCGAN object.
        This function is intend to record the trained model and generated anime faces of this model during every 100 epoch of training.
        '''
        r,c = 5, 5
        fig, axes = plt.subplots(r, c, figsize = (10, 10))
        fig.subplots_adjust(hspace = 0.5, wspace = 0.5)

        noise = np.random.normal(0, 1, (r * c, self.noise_dim))
        img_gener_ = self.generator.predict(noise)
        img_gener_ = 0.5 * img_gener_+ 0.5
        for i, ax in enumerate(axes.flat):
            ax.imshow(img_gener_[i,:,:,:])
            ax.set_xticks([])
            ax.set_yticks([])
        if tags is None:  
            if iteration < epochs - 1:
                model_address = os.path.join(self.model_store_address, "model_gans_{0}".format(iteration))
                image_address = os.path.join(self.model_store_address, "model_{0}.png".format(iteration))
                self.combine.save(model_address)
                fig.savefig(image_address)
            else:
                model_address = os.path.join(self.model_store_address, "model_gans_final_{0}".format(iteration))
                image_address = os.path.join(self.model_store_address, "model_final_{0}.png".format(iteration))
                self.combine.save(model_address)
                fig.savefig(image_address)

        else:
            if iteration < epochs - 1:
                model_address = os.path.join(self.model_store_address, "model_gans_{0}_{1}".format(tags, iteration))
                image_address = os.path.join(self.model_store_address, "model_{0}_{1}.png".format(tags, iteration))
                self.combine.save(model_address)
                fig.savefig(image_address)
            else:
                 model_address = os.path.join(self.model_store_address, "model_gans_final_{0}_{1}".format(tags, iteration))
                image_address = os.path.join(self.model_store_address, "model_final_{0}_{1}.png".format(tags, iteration))
                self.combine.save(model_address)
                fig.savefig(image_address)
        plt.show()
        plt.close()
        
    def update_(self, generate_, images_, threhold):
        '''
        This function is used to tuning the complete between G and D during training. 
        If you can't find the right balanced between them, try this function.
        Typically,
        if True is returned, D could be trained in this epoch/iteration(depend on which for level in train() you put in.).
        else, D would be freezed in this epoch/iteration(depend on which for level in train() you put in.).
        '''
        pred_ = self.discriminator.predict(generate_)
        cls_ = self.discriminator.predict(images_)

        pred_true = (pred_ < 0.5).astype(int)
        cls_true = (cls_ >= 0.5).astype(int)

        acc_ = (np.sum(pred_true)/len(generate_) + np.sum(cls_true)/len(images_))/2.
        if acc_ < threhold:
            return True
        else:
            return False

    def train(self, epochs, data, noise = True, tags = None, batch_size = 32, print_interval = 100, save_interval = 1):
        '''
        epochs(int): # of epochs for whole training process.
        data(np.array): dataset including anime faces during training.
        noise(bool) whether all noise to train data during training.
        tags: tag name.
        
        This function is used to train the complied model defined in __init__().
        '''
        #change the data here...
        #train_data = np.expand_dims(train_data, axis = 3)

        valid = np.ones((batch_size, 1), dtype = np.float32)
        fake = np.zeros((batch_size, 1), dtype = np.float32)
        loss_d = 0
        acc_d = 0
        print("start training")
        for epoch in range(epochs):
            train_data = self.shuffle_data(data)
            for i in range(int(train_data.shape[0]/batch_size)):
                images_ = train_data[i*batch_size:(i+1)*batch_size]
                if noise:
                    images_ = self.add_noise(data_read_ = images_, mean = 1e-2, std = 1e-2)
                noise_images_ = np.random.normal(0, 1, (batch_size, self.noise_dim))
                generated_ = self.generator.predict(noise_images_)
                d_r = self.discriminator.train_on_batch(generated_, fake)
                d_f = self.discriminator.train_on_batch(images_, valid)
                loss_d = 1/2. * (d_r[0] + d_f[0])
                acc_d = 1/2. * (d_r[1] + d_f[1])

                _ = self.combine.train_on_batch(noise_images_, valid)
                d_combine = self.combine.train_on_batch(noise_images_, valid)
                if i % print_interval == 0:
                    try:
                        print("Under epoch {0}: iteration -> {1}, the Loss of discriminator is {2:.3}, and accuracy is {3:.2%}, the Loss of generator is {4:.3}".format(epoch, i, loss_d, acc_d, d_combine[0]))
                    except:
                        print("Under epoch ", epoch, ": iteration -> ", i, ", the Loss of discriminator is ", loss_d, ", and accuracy is ", acc_d, ", the Loss of generator is ", d_combine[0])

            if epoch % save_interval == 0 or epoch == epochs - 1:
                self.save_and_plot_images(epoch, tags = tags, epochs = epochs)
