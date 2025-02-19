import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# data_folder = insert absolute path of "cifar-10-batches-py"
data_folder = 'C:/Hofa/ComVis/Exercises/Proj2.1/cifar-10-batches-py'
def training_set():
    training_images = []
    training_labels = []

    for batch in range(5):
        file = f'{data_folder}/data_batch_{batch + 1}'
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

    file = f'{data_folder}/test_batch'
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

# training_images, training_labels = training_set()
# # print(training_images.shape)
# # print(training_labels.shape)

# test_images, test_labels = test_set()
# # print(test_images.shape)
# # print(test_labels.shape)

with open(f'{data_folder}/batches.meta', 'rb') as fo:
    info = pickle.load(fo, encoding='bytes')
label_names = info[b'label_names']

def visualize(image, label):
    label = label_names[label]
    plt.figure(figsize=(2, 2))
    plt.subplot(1, 1, 1), plt.imshow(image), plt.title(label)
    plt.show()
    return

# i = 20
# visualize(training_images[i], training_labels[i])

def calculate_hu_moments(image_batch):
    hu_moments_batch = []

    for image in image_batch:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        moments = cv2.moments(gray_image)
        hu_moments = cv2.HuMoments(moments).flatten()
        for i in range(0,7):
            hu_moments[i] = -1* math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))
        hu_moments_batch.append(hu_moments)

    hu_moments_batch = np.array(hu_moments_batch)

    return hu_moments_batch

# hu_moments_batch = calculate_hu_moments(training_images)
# print(hu_moments_batch.shape)

def compute_phog(image, num_bins=8, pyramid_levels=3):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    bin_edges = np.linspace(0, 360, num_bins + 1)
    angle_bins = np.digitize(angle, bin_edges) - 1

    phog_features = []

    for level in range(pyramid_levels):
        num_cells = 2 ** level
        cell_height = gray_image.shape[0] // num_cells
        cell_width = gray_image.shape[1] // num_cells

        for i in range(num_cells):
            for j in range(num_cells):
                cell_magnitude = magnitude[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                cell_angle_bins = angle_bins[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]

                hist = np.zeros(num_bins, dtype=np.float32)
                for b in range(num_bins):
                    hist[b] = np.sum(cell_magnitude[cell_angle_bins == b])

                hist /= (np.sum(hist) + 1e-6)
                phog_features.extend(hist)
    return np.array(phog_features)

def calculate_phog_batch(image_batch, num_bins=8, pyramid_levels=3):
    phog_batch = []
    for image in image_batch:
        phog_features = compute_phog(image, num_bins, pyramid_levels)
        phog_batch.append(phog_features)

    phog_batch = np.array(phog_batch)
    return phog_batch

# phog_batch = calculate_phog_batch(training_images)
# print(phog_batch.shape)
