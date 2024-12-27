import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

def training_set():
    training_images = []
    training_labels = []

    for batch in range(5):
        file = f'cifar-10-batches-py/data_batch_{batch + 1}'
        with open(file, 'rb') as fo:
            data_batch = pickle.load(fo, encoding='bytes')

        image_batch_raw = data_batch[b'data']
        labels_batch = data_batch[b'labels']

        for id, image in enumerate(image_batch_raw):
            red_channel = image[:1024].reshape(32, 32)     # Red values
            green_channel = image[1024:2048].reshape(32, 32)  # Green values
            blue_channel = image[2048:].reshape(32, 32)    # Blue values

            # Stack the channels along the last axis to create a (32, 32, 3) image
            final_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

            # Append the reshaped image to the training set
            training_images.append(final_image)
            training_labels.append(labels_batch[id])
    training_images = np.stack(training_images, axis=0)
    training_labels = np.stack(training_labels, axis=0)
    return training_images, training_labels

def test_set():
    training_images = []
    training_labels = []

    file = f'cifar-10-batches-py/test_batch'
    with open(file, 'rb') as fo:
        data_batch = pickle.load(fo, encoding='bytes')

    image_batch_raw = data_batch[b'data']
    labels_batch = data_batch[b'labels']

    for id, image in enumerate(image_batch_raw):
        red_channel = image[:1024].reshape(32, 32)     # Red values
        green_channel = image[1024:2048].reshape(32, 32)  # Green values
        blue_channel = image[2048:].reshape(32, 32)    # Blue values

        # Stack the channels along the last axis to create a (32, 32, 3) image
        final_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

        # Append the reshaped image to the training set
        training_images.append(final_image)
        training_labels.append(labels_batch[id])
    training_images = np.stack(training_images, axis=0)
    training_labels = np.stack(training_labels, axis=0)
    return training_images, training_labels

training_images, training_labels = training_set()
# print(training_images.shape)
# print(training_labels.shape)

test_images, test_labels = test_set()
# print(test_images.shape)
# print(test_labels.shape)


with open("cifar-10-batches-py/batches.meta", 'rb') as fo:
    info = pickle.load(fo, encoding='bytes')
label_names = info[b'label_names']

def visualize(image, label):
    label = label_names[label]
    plt.figure(figsize=(2, 2))
    plt.subplot(1, 1, 1), plt.imshow(image), plt.title(label)
    plt.show()
    return

i = 15
visualize(training_images[i], training_labels[i])

