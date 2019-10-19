import os
import random

import keras
from keras import layers
from keras.preprocessing import image
import numpy as np
import utils

print("keras version: " + keras.__version__)

save_folder_base_name = os.path.basename(__file__)
image_sample_name = 'simpson_faces'
description = 'no_dropout_ReLU'

image_shape = (64, 64, 3)
latent_dim = 100
channels = 3
iterations = 10000
batch_size = 128
epsilon = 0.00005
weight_init_stddev = 0.02
beta_1 = 0.5
discriminator_lr = 0.0004
gan_lr = 0.0004
momentum = 0.9

kernel_initializer = keras.initializers.RandomNormal(
    mean=0, stddev=weight_init_stddev)

gamma_initializer = keras.initializers.RandomNormal(
    mean=1, stddev=weight_init_stddev)

save_key_pick_list = [
    'save_folder_base_name', 'description', 'gan_lr', 'discriminator_lr'
]

hyper_params = [
    str(item) for key, item in dict.copy(globals()).items() if
    key in save_key_pick_list
]

dir_name = os.path.dirname(__file__)
image_dir = os.path.join(dir_name, 'images')
save_dir = os.path.join(dir_name, 'samples', "_".join(hyper_params))

generator_save_path = os.path.join(save_dir, 'generator.h5')
discriminator_save_path = os.path.join(save_dir, 'discriminator.h5')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

datagen = image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)

batches = datagen.flow_from_directory(image_dir,
                                      target_size=(image_shape[0:2]),
                                      batch_size=batch_size,
                                      class_mode=None,
                                      classes=[image_sample_name])


def Discriminator(use_bias=False):
    discriminator = keras.Sequential([

        layers.Conv2D(filters=64, kernel_size=[5, 5], input_shape=image_shape,
                      padding='SAME', strides=(2, 2), use_bias=use_bias),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(filters=128, kernel_size=[5, 5],
                      padding='SAME', strides=(2, 2), use_bias=use_bias),
        layers.BatchNormalization(epsilon=epsilon, momentum=momentum,
                                  gamma_initializer=gamma_initializer),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(filters=256, kernel_size=[5, 5],
                      padding='SAME', strides=(2, 2), use_bias=use_bias),

        layers.BatchNormalization(epsilon=epsilon, momentum=momentum,
                                  gamma_initializer=gamma_initializer),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(filters=512, kernel_size=[5, 5],
                      padding='SAME', strides=(2, 2), use_bias=use_bias),

        layers.BatchNormalization(epsilon=epsilon, momentum=momentum,
                                  gamma_initializer=gamma_initializer),
        layers.LeakyReLU(alpha=0.2),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')

    ])

    discriminator.summary()
    return discriminator


def Generator(use_bias=False):
    generator = keras.Sequential([

        layers.Dense(1024 * 4 * 4, input_shape=(latent_dim,)),
        layers.ReLU(),
        layers.Reshape((4, 4, 1024)),

        layers.Conv2DTranspose(filters=512, kernel_size=[5, 5],
                               data_format='channels_last',
                               padding='SAME', strides=(2, 2),
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer),
        layers.BatchNormalization(epsilon=epsilon, momentum=momentum,
                                  gamma_initializer=gamma_initializer),
        layers.ReLU(),
        layers.Conv2DTranspose(filters=256, kernel_size=[5, 5],
                               data_format='channels_last',
                               padding='SAME', strides=(2, 2),
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer),
        layers.BatchNormalization(epsilon=epsilon, momentum=momentum,
                                  gamma_initializer=gamma_initializer),
        layers.ReLU(),
        layers.Conv2DTranspose(filters=128, kernel_size=[5, 5],
                               data_format='channels_last',
                               padding='SAME', strides=(2, 2),
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer),
        layers.BatchNormalization(epsilon=epsilon, momentum=momentum,
                                  gamma_initializer=gamma_initializer),
        layers.ReLU(),
        layers.Conv2DTranspose(filters=64, kernel_size=[5, 5],
                               data_format='channels_last',
                               padding='SAME', strides=(2, 2),
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer),
        layers.BatchNormalization(epsilon=epsilon, momentum=momentum,
                                  gamma_initializer=gamma_initializer),
        layers.ReLU(),
        layers.Conv2DTranspose(filters=32, kernel_size=[5, 5],
                               data_format='channels_last',
                               padding='SAME', strides=(1, 1),
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer),
        layers.BatchNormalization(epsilon=epsilon, momentum=momentum,
                                  gamma_initializer=gamma_initializer),
        layers.ReLU(),
        layers.Conv2DTranspose(filters=3, kernel_size=[5, 5],
                               data_format='channels_last',
                               padding='SAME', strides=(1, 1),
                               activation='tanh')
    ])

    generator.summary()
    return generator


generator = Generator()
discriminator = Discriminator()

if os.path.isfile(discriminator_save_path):
    discriminator.load_weights(discriminator_save_path)

if os.path.isfile(generator_save_path):
    generator.load_weights(generator_save_path)

# Use gradient clipping (by specifying a value) in the optimizer
# Use learning rate attenuation for stable training
discriminator_optimizer = keras.optimizers.Adam(lr=discriminator_lr,
                                                beta_1=beta_1)
discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy')

# Set discriminator's weight not trained (applies to gan models only)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.Adam(lr=gan_lr, beta_1=beta_1)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Start training repetition
epoch = 0

for index, batch in enumerate(batches):

    batch_size = batch.shape[0]
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    real_images = batch

    # create fake image
    fake_images = generator.predict(random_latent_vectors)

    # !important, add noise to real image. ( It is really helpful. )
    real_images += np.random.normal(0, random.uniform(0.0, 0.02),
                                    size=real_images.shape)

    # create real ones with one and generated ones with zeros labels.
    # add random noise to labels. it's very important.
    real_labels = np.ones((batch_size, 1)) + 0.05 * np.random.random((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1)) + 0.05 * np.random.random((batch_size, 1))

    # train discriminator
    discriminator.trainable = True
    d_real_loss = discriminator.train_on_batch(real_images, real_labels)
    d_fake_loss = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * (d_real_loss + d_fake_loss)

    # create latent vectors with random value.
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # set the labels as real ones to fool discriminator.
    misleading_targets = np.ones((batch_size, 1))

    # train generator (in gan model, discriminator's weights will be frozen.)
    discriminator.trainable = False
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    if index % len(batches) == 0:
        epoch += 1

        generator.save_weights(generator_save_path)
        discriminator.save_weights(discriminator_save_path)

        # Output the metrics
        print('Discriminator loss in step %s: %s' % (epoch, d_loss))
        print('Gan loss in step %s: %s' % (epoch, a_loss))

        # Save one created image
        fakes = generator.predict(
            np.random.normal(size=(5, latent_dim)))

        utils.save_sample_images(
            fakes, save_dir, str(epoch).zfill(3))

        # Save one real image for comparison
        real = image.array_to_img(real_images[0])
        real.save(os.path.join(save_dir, 'real_image' + str(epoch) + '.png'))

        real_result = np.mean(
            discriminator.predict(real_images))

        fake_result = np.mean(
            discriminator.predict(fake_images))
        print('Discriminator real image prediction result: %s' % str(real_result))
        print('Discriminator fake image prediction result: %s' % str(fake_result))
