from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
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
    #  print(roi_data.shape)
    plt.imshow(response_data[0])
    plt.show()
    test_image = np.where(roi_data==0,  0, response_data)
    print(test_image.shape)
    plt.imshow(test_image[2])
    plt.show()
    print(np.load("stimparams.dict", allow_pickle=True))
# load_response_data('response_array_1s_interval.npy', '1_roi_morphed.npy')
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
            print(image_number, tif_file)
            # Load the TIFF image
            # print('Loading:', tif_file, 'Image number:', image_number)
            tif_image = Image.open(os.path.join('Images', tif_file)).convert("L")
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
    highlow_data[np.logical_not(mask)] = np.nan
    # highlow_data[np.logical_not(mask)] = 0
    return highlow_data

def match_pictures_with_response(stims_param_name="stimparams.dict", roi_data_name="1_roi_morphed.npy", response_data_name="response_array_1s_interval.npy"):
    stims_shown = np.load(stims_param_name, allow_pickle=True)
    picture_ids = [int(t[0]) for t in stims_shown["stimDuration"][:]]
    print(picture_ids)
    print(len(picture_ids))
    print(max(picture_ids))
    print(min(picture_ids))
    # print(picture_ids)
    roi_data = np.load(roi_data_name)
    response_data = np.load(response_data_name).squeeze()
    # response_data = np.where(roi_data == 0, 0, response_data)
    resolution = 1000/182
    plt.imshow(response_data[100])
    plt.title("Original Response before normalizing")
    plt.colorbar()
    plt.show()
    print(response_data[0].shape)
    print(response_data[0])
    # response_data = lowhigh_normalize(frame=response_data, mask=roi_data, sig_high=resolution, sig_low=100)
    images_responses = []
    # print(response_data.shape)
    # for id, tif_file in enumerate(tif_files):
    #     curr_image = Image.open(os.path.join('Images', tif_file)).convert("L")
    #     curr_image = np.array(curr_image)
    #     if curr_image.shape != (1920, 2560):
    #         print(tif_file, curr_image.shape)
    #         continue
    #     images[id] = curr_image
    print(f"{response_data.shape} response data shape")
    #     images_respones.append([curr_image, response_data[id - 1]])
    for i, id in enumerate(picture_ids):
        #print(i, id)
        if id < 151:
            curr_picture = find_image(id-1)
            if curr_picture is not None:
                if curr_picture.shape != (1920, 2560):
                    print(curr_picture.shape, "Not right format")
                    continue
                images_responses.append([curr_picture, response_data[i-1]])
        else:
            print(i, id)
    print(f"{len(images_responses)}, len image responses")
    images = np.zeros([147, 1920, 2560])
    responses_images = np.zeros([147, 270, 320]) # fix with mask
    for j, tuple in enumerate(images_responses):
        images[j] = tuple[0]
        responses_images[j] = tuple[1]
    plt.imshow(response_data[10])
    plt.title("Original Response")
    plt.colorbar()
    plt.show()
    plt.imshow(images[0])
    plt.show()
    print(len(responses_images), images.shape)
    # Remove None values
    print(images_responses[0][0].shape, images_responses[0][1].shape)
    # for counter, tuple in enumerate(images_respones):
    #     if tuple[0] is None:
    #         # print(tuple[0].shape, tuple[1].shape)
    #         images_respones.pop(counter)
    # print(len(images_respones))
    return images_responses, images, responses_images

# images_responses, images, responses = match_pictures_with_response()
import pickle as pk
def apply_pca(data, n_components, response_data=None, mask=None, pca_vector=None):
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
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    print(f"Original data shape: {data.shape}, Transformed data shape: {transformed_data.shape}")
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(cumulative_variance)
    print(cumulative_variance.shape)
    reconstructed_data = pca.inverse_transform(transformed_data)
    reconstructed_data = scaler.inverse_transform(reconstructed_data)
    if pca_vector is not None:
        # Reconstruct a single response from its PCA vector
        single_reconstructed = pca.inverse_transform(pca_vector.reshape(1, -1))
        single_reconstructed = scaler.inverse_transform(single_reconstructed)
        single_reconstructed_response = np.zeros(mask.shape)
        single_reconstructed_response[mask] = single_reconstructed.flatten()
        single_reconstructed_response[np.logical_not(mask)] = np.nan
        plt.imshow(single_reconstructed_response, cmap='gray')
        plt.title("Reconstructed Single Response from PCA Vector")
        plt.colorbar()
        plt.show()
    if response_data:
        # print(reconstructed_data.shape)
        reshaped_responses = np.zeros((147, 270* 320))
        reshaped_responses[:, mask.flatten()] = reconstructed_data
        reconstructed_responses = reshaped_responses.reshape(147, 270, 320)
        reconstructed_responses[:, np.logical_not(mask)] = np.nan
        plt.imshow(reconstructed_responses[10], cmap='gray')
        plt.colorbar()
        plt.title("Reconstructed Response After PCA")
        plt.show()
        pk.dump(pca, open("pca_responses.pkl", "wb"))
        pk.dump(scaler, open("scaler_responses.pkl", "wb"))
    else:
        reconstructed_images = reconstructed_data.reshape(147, 1920, 2560)
        print(reconstructed_images.shape)
        plt.imshow(reconstructed_images[0])
        plt.title("Reconstructed Image(input) After PCA")
        plt.show()
        pk.dump(pca, open("pca_images.pkl", "wb"))
        pk.dump(scaler, open("scaler_images.pkl", "wb"))
    return transformed_data

# only select mask area:
# roi_mask = np.load("1_roi_morphed.npy")
# responses = np.array(responses).reshape(responses.shape[0], -1)[:, roi_mask.flatten()]
# print(responses.shape)
# response_data_pca = apply_pca(data=responses, n_components=15, response_data=True, mask=roi_mask, pca_vector=None)
# response_data_pca = apply_pca(data=responses, n_components=15, response_data=True, mask=roi_mask, pca_vector=response_data_pca[10])
# np.save("responses_no_normalize_pca.npy", response_data_pca)
# print(images.shape)
# images = images.reshape(images.shape[0], 1920*2560)
# images_pca = apply_pca(data=images, n_components=130, response_data=False)
# np.save("images_pca.npy", images_pca)
# 95 mit 102 aber kann das bild nicht mehr erkennen
# 147 noch 100% variance

import torch
import numpy as np
import torch.nn as nn
def test_model(model_name, input_image, true_output, pca_name_response, scaler_name_responses, mask):
    pca_reloaded_responses = pk.load(open(pca_name_response, 'rb'))
    scaler_reloaded_responses = pk.load(open(scaler_name_responses, 'rb'))
    model = PopulationCNN(input_size=130, output_size=15)
    model.to("cpu")
    model.load_state_dict(torch.load(model_name, weights_only=True))
    model.eval()
    # print(f"{input_image.shape}, input_shape") # should be 1, 1, 130
    # input_image = input_image.astype(np.float32).reshape(-1, 1, input_image.shape[1])
    input_image = input_image.astype(np.float32).reshape(-1, 1, input_image.shape[0])
    # print(f"{input_image.shape}, input_shape")
    input_image = torch.tensor(input_image)
    # print(f"{input_image.shape}, input_shape")
    with torch.no_grad():
        output_image = model(input_image)
    # print(f"{output_image.shape}, output_shape") # should be 1, 147
    true_output = torch.tensor(true_output.reshape(1, 15)) # second shape is number of pca components
    # print(f"{true_output.shape}, true output shape")  # should be 1, 147
    loss = nn.MSELoss()
    print(true_output[0])
    print(output_image[0])
    print(f"loss {loss(output_image, true_output)}")
    single_reconstructed = pca_reloaded_responses.inverse_transform(output_image.reshape(1, -1))
    single_reconstructed = scaler_reloaded_responses.inverse_transform(single_reconstructed)
    single_reconstructed_response = np.zeros(mask.shape)
    single_reconstructed_response[mask] = single_reconstructed.flatten()
    single_reconstructed_response[np.logical_not(mask)] = np.nan
    plt.imshow(single_reconstructed_response)
    plt.title("Model Output Reconstructed from PCA Vector")
    plt.colorbar()
    plt.show()
    single_reconstructed2 = pca_reloaded_responses.inverse_transform(true_output.reshape(1, -1))
    single_reconstructed2 = scaler_reloaded_responses.inverse_transform(single_reconstructed2)
    single_reconstructed_response2 = np.zeros(mask.shape)
    single_reconstructed_response2[mask] = single_reconstructed2.flatten()
    single_reconstructed_response2[np.logical_not(mask)] = np.nan
    plt.imshow(single_reconstructed_response2)
    plt.title("True Output Reconstructed from PCA Vector")
    plt.colorbar()
    plt.show()



    return output_image
from model import PopulationCNN
responses_pca = np.load("responses_no_normalize_pca.npy")
images_pca = np.load("images_pca.npy")

# TODO split in test and train
# TODO clean up code
#


# test_model("trained_model_weights_500.pth", images_pca[10], responses_pca[10], "pca_responses.pkl", "scaler_responses.pkl", np.load("1_roi_morphed.npy"))
test_model("trained_model_weights_250.pth", images_pca[10], responses_pca[10], "pca_responses.pkl", "scaler_responses.pkl", np.load("1_roi_morphed.npy"))