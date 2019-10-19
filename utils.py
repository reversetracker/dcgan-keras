import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from keras.preprocessing import image


def load_data(tf, file_path=None, image_width=64, image_height=64, num_parallel_calls=20, batch_size=100, buffer_size=1):
  file_paths = [join(file_path, f) for f in listdir(file_path) if
                isfile(join(file_path, f))]

  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  # Source
  def load_image(path):
    image_string = tf.read_file(path)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # image = tf.image.resize_images(image, [image_width, image_height])
    image = tf.image.convert_image_dtype(image, tf.float32)
    # This will convert to float values in [0, 1]
    image /= 127.5
    image -= 1

    return image

  # Apply the function load_image to each filename in the dataset
  dataset = dataset.map(load_image, num_parallel_calls=num_parallel_calls)

  # Create batches images each
  dataset = dataset.shuffle(buffer_size=batch_size)
  dataset = dataset.batch(batch_size)
  # dataset = dataset.repeat(10)

  dataset = dataset.prefetch(buffer_size=buffer_size)

  iterator = dataset.make_initializable_iterator()

  return iterator, len(file_paths)

def save_sample_images(samples, folder_path, name):
  sample_size = samples.shape[0]

  fig, ax = plt.subplots(1, sample_size)

  for i in range(sample_size):
    ax[i].set_axis_off()
    ax[i].imshow(image.array_to_img(samples[i]))

  plt.savefig(os.path.join(folder_path, '{}.png'.format(name)),
              bbox_inches='tight')
  plt.close(fig)
