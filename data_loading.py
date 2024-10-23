import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as snd

def load_npy(data_name):
    # Load the .npy file
    data = np.load(data_name)

    # Now 'data' is a NumPy array containing the data from the .npy file
    print('Data shape:', data.shape)
    # print(data)
    plt.imshow(data)
    plt.show()
    # Loop through the first 10 items in data[0]
    # for counter, i in enumerate(data[0]):
    #     if counter < 10:
    #         print('Item shape:', i.shape, 'Counter:', counter)
    #
    #         # Extract the image
    #         image = data[0][counter][0]
    #         print('Image shape:', image.shape)
    #
    #         # Compute the mean and standard deviation for the current image
    #         # MU = np.mean(image)
    #         # SD = np.std(image)
    #         # print(f'Image {counter} - Mean: {MU}, Std: {SD}')
    #
    #         # Display the image using imshow
    #         # plt.imshow(
    #         #     image,
    #         #     interpolation="nearest",
    #         #     cmap="binary_r",
    #         #     vmin=MU - 2 * SD,
    #         #     vmax=MU + 2 * SD
    #         # )
    #         plt.imshow(image)
    #         plt.colorbar()
    #         plt.title(f'Image {counter}')
    #         plt.show()

def load_dict(data_name):
    # Replace 'your_data.npy' with the path to your .npy file
    data = np.load(data_name, allow_pickle=True)
    print(data)

np.set_printoptions(threshold=np.inf)
def load_response_data(response_data_name, roi_data_name):
    response_data = np.load(response_data_name).squeeze()
    print('Data shape:', response_data.shape)
    roi_data = np.load(roi_data_name)
    plt.imshow(response_data[0])
    plt.show()
    test_image = np.where(roi_data==0,  0, response_data)
    print(test_image.shape)
    plt.imshow(test_image[2])
    plt.show()
    print(np.load("stimparams.dict", allow_pickle=True))
# load_response_data('response_array_1s_interval.npy', '1_roi_morphed.npy')
from PIL import Image

import os
from PIL import Image
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
import matplotlib.pyplot as plt


import os
import re
from PIL import Image
import matplotlib.pyplot as plt
def load_images(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter out only TIFF files
    tif_files = [f for f in files if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    image_numbers = []
    for counter, tif_file in enumerate(tif_files):
        if counter < 250:
            # Extract digits from the file name
            image_number = re.findall(r'\d+', tif_file)
            image_number = image_number[0] if image_number else 'No number'
            image_number = int(image_number)
            if image_number not in image_numbers:
                # print(image_number, tif_file, 'is not a duplicate')
                image_numbers.append(image_number)
            else:
                print(image_number, tif_file, 'is a duplicate')
            # Load the TIFF image
            # print('Loading:', tif_file, 'Image number:', image_number)
            # tif_image = Image.open(os.path.join(directory, tif_file))
            # Display the image using matplotlib
            # plt.imshow(tif_image)
            # plt.axis('off')  # Hide the axis
            # plt.title(f'{tif_file} (Image number: {image_number})')
            # plt.show()
    # print(set(image_numbers))
    # print(find_duplicates(image_numbers))
    # stims_shown = np.load("stimparams.dict", allow_pickle=True)
    # first_values = [int(t[0]) for t in stims_shown["stimDuration"][:]]
    # intersection = set(first_values).intersection(set(image_numbers))
    # print(intersection)


def find_image(number):
    # List all files in the directory
    files = os.listdir('Images')

    # Filter out only TIFF files
    tif_files = [f for f in files if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    for tif_file in tif_files:
        # Extract digits from the file name
        image_number = re.findall(r'\d+', tif_file)
        image_number = image_number[0] if image_number else 'No number'
        image_number = int(image_number)
        if image_number == number:
            # Load the TIFF image
            # print('Loading:', tif_file, 'Image number:', image_number)
            tif_image = Image.open(os.path.join('Images', tif_file))
            # Display the image using matplotlib
            # plt.imshow(tif_image)
            # plt.axis('off')  # Hide the axis
            # plt.title(f'{tif_file} (Image number: {image_number})')
            # plt.show()
            return np.array(tif_image)
    # print("Image not found", number)
    return None


def lowhigh_normalize(frame, mask=None, sig_high=None, sig_low=None):
    ''' apply bandpass filter to frame
    specify mask and standard deviations sig_high (highpass) and sig_low (lowpass) of gaussian filters
    '''
    # recursion for frames
    if len(frame.shape) > 2:
        result = np.empty_like(frame)
        for i in range(frame.shape[0]):
            result[i] = lowhigh_normalize(frame[i], mask, sig_high, sig_low)
        return result

    # recursion for complex images
    if np.iscomplexobj(frame):
        result = np.empty_like(frame)
        result.real = lowhigh_normalize(frame.real, mask, sig_high, sig_low)
        result.imag = lowhigh_normalize(frame.imag, mask, sig_high, sig_low)
        return result

    if mask is None:
        mask = np.ones(frame.shape, dtype=np.bool)
    data = np.copy(frame)
    m = np.zeros(mask.shape)
    m[mask] = 1.0
    m[np.logical_not(np.isfinite(frame))] = 0.
    data[np.logical_not(m)] = 0.
    m2 = np.copy(m)

    ## gaussian low pass
    low_mask = snd.gaussian_filter(m2, sig_low, mode='constant', cval=0)
    low_data = 1. * snd.gaussian_filter(data, sig_low, mode='constant', cval=0) / low_mask

    #### linear low pass
    ##t = np.linspace(-1,1,11)
    ##kernel = t.reshape(11,1)*t.reshape(1,11)
    ##low_mask = snd.convolve(m2, kernel, mode='constant', cval=0.0)
    ##low_data = snd.convolve(data, kernel, mode='constant', cval=0.0)/low_mask

    low_data[np.logical_not(m)] = 0
    high_mask = snd.gaussian_filter(m, sig_high, mode='constant', cval=0)
    highlow_data = low_data - 1. * snd.gaussian_filter(low_data, sig_high, mode='constant', cval=0) / high_mask
    # highlow_data[np.logical_not(mask)] = np.nan
    highlow_data[np.logical_not(mask)] = 0
    return highlow_data

def match_pictures_with_response(stims_param_name="stimparams.dict", roi_data_name="1_roi_morphed.npy", response_data_name="response_array_1s_interval.npy"):
    stims_shown = np.load(stims_param_name, allow_pickle=True)
    # picture_ids = [int(t[0]) for t in stims_shown["stimDuration"][:]]
    # print(picture_ids)
    roi_data = np.load(roi_data_name)
    response_data = np.load(response_data_name).squeeze()
    # response_data = np.where(roi_data == 0, 0, response_data)
    resolution = 1000/182
    response_data = lowhigh_normalize(frame=response_data, mask=roi_data, sig_high=resolution, sig_low=100)
    images_respones = []
    files = os.listdir('Images')
    tif_files = [f for f in files if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    tif_files.sort()
    images = np.zeros([150, 1920, 2560])
    # print(response_data.shape)
    for id, tif_file in enumerate(tif_files):
        curr_image = Image.open(os.path.join('Images', tif_file)).convert("L")
        curr_image = np.array(curr_image)
        if curr_image.shape != (1920, 2560):
            print(tif_file, curr_image.shape)
            continue
        images[id] = curr_image
        images_respones.append([curr_image, response_data[id - 1]])
    # for i in picture_ids:
    #     if i < 151:
    #         # curr_picture = find_image(i)
    #         images_respones.append([curr_picture, response_data[i-1]])
    #
    plt.imshow(response_data[0], cmap='gray')
    plt.colorbar()
    plt.show()
    plt.imshow(images[0])
    plt.show()
    print(len(images_respones))
    # Remove None values
    print(images_respones[0][0].shape, images_respones[0][1].shape)
    # for counter, tuple in enumerate(images_respones):
    #     if tuple[0] is None:
    #         # print(tuple[0].shape, tuple[1].shape)
    #         images_respones.pop(counter)
    # print(len(images_respones))
    return images_respones, images, response_data

images_responses, images, response_data = match_pictures_with_response()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def apply_pca(data, n_components, response_data=None):
    """
    Apply PCA to the given data.

    Parameters:
    data (numpy.ndarray): The input data array.
    n_components (int): The number of principal components to keep.

    Returns:
    numpy.ndarray: The transformed data after applying PCA.
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # Impute NaN values with the mean of each feature
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    print(f"Original data shape: {data.shape}, Transformed data shape: {transformed_data.shape}")
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(cumulative_variance)
    print(cumulative_variance.shape)
    reconstructed_data = pca.inverse_transform(transformed_data)
    reconstructed_data = scaler.inverse_transform(reconstructed_data)
    if response_data:
        reconstructed_images = reconstructed_data.reshape(150, 270, 320)
        print(reconstructed_images.shape)
        plt.imshow(reconstructed_images[0], cmap='gray')
        plt.colorbar()
        plt.title("Reconstructed Image(output) After PCA")
        plt.show()
    else:
        reconstructed_images = reconstructed_data.reshape(150, 1920, 2560)
        print(reconstructed_images.shape)
        plt.imshow(reconstructed_images[0])
        plt.title("Reconstructed Image(input) After PCA")
        plt.show()
    return transformed_data

# Example usage:
# print(response_data[0].shape)
# print(response_data.shape)
# response_data = response_data.reshape(response_data.shape[0], -1)
# response_data_pca = apply_pca(data=response_data, n_components=100, True)
print(images.shape)
images = images.reshape(images.shape[0], 1920*2560)
images_pca = apply_pca(data=images, n_components=102, response_data=False)
# 95 mit 102 aber kann das bild nicht mehr erkennen