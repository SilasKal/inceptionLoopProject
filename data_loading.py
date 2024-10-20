import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


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
load_response_data('response_array_1s_interval.npy', '1_roi_morphed.npy')
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

def match_pictures_with_response(stims_param_name="stimparams.dict", roi_data_name="1_roi_morphed.npy", response_data_name="response_array_1s_interval.npy"):
    stims_shown = np.load(stims_param_name, allow_pickle=True)
    picture_ids = [int(t[0]) for t in stims_shown["stimDuration"][:]]
    print(picture_ids)
    roi_data = np.load(roi_data_name)
    response_data = np.load(response_data_name).squeeze()
    response_data = np.where(roi_data == 0, 0, response_data)
    images_respones = []
    for i in picture_ids:
        if i < 151:
            curr_picture = find_image(i)
            images_respones.append([curr_picture, response_data[i-1]])
    # plt.imshow(images_respones[75][0])
    # plt.show()
    # plt.imshow(images_respones[75][1])
    # plt.show()
    print(len(images_respones))
    # Remove None values
    for counter, tuple in enumerate(images_respones):
        if tuple[0] is None:
            # print(tuple[0].shape, tuple[1].shape)
            images_respones.pop(counter)
    print(len(images_respones))
    return images_respones

match_pictures_with_response()
# Example usage
# load_images('Images')

# load_npy('response_array_1s_interval.npy')
# load_npy('stimulus.npy')
# load_dict("stimparams.dict")
# load_npy("1_roi_morphed.npy")
# load_response_data('response_array_1s_interval.npy', '1_roi_morphed.npy')
