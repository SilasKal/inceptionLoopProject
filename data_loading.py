from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import re
from PIL import Image

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
    test_image = np.where(roi_data == 0, 0, response_data)
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
            # print(image_number, tif_file)
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
    """ apply bandpass filter to frame
    specify mask and standard deviations sig_high (highpass) and sig_low (lowpass) of gaussian filters
    """
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

    # gaussian low pass
    low_mask = snd.gaussian_filter(m2, sig_low, mode='constant', cval=0)
    low_data = 1. * snd.gaussian_filter(data, sig_low, mode='constant', cval=0) / low_mask

    # linear low pass
    # t = np.linspace(-1,1,11)
    # kernel = t.reshape(11,1)*t.reshape(1,11)
    # low_mask = snd.convolve(m2, kernel, mode='constant', cval=0.0)
    # low_data = snd.convolve(data, kernel, mode='constant', cval=0.0)/low_mask
    # print(m, "mask")
    # print(sig_high, sig_low)
    low_data[np.logical_not(m)] = 0
    high_mask = snd.gaussian_filter(m, sig_high, mode='constant', cval=0)
    # print(high_mask)
    highlow_data = low_data - 1. * snd.gaussian_filter(low_data, sig_high, mode='constant', cval=0) / high_mask
    highlow_data[np.logical_not(mask)] = np.nan
    # highlow_data[np.logical_not(mask)] = 0
    return highlow_data

def match_pictures_with_response(stims_param_name="stimparams.dict", roi_data_name="1_roi_morphed.npy",
                                 response_data_name="response_array_1s_interval.npy"):
    stims_shown = np.load(stims_param_name, allow_pickle=True)
    picture_ids = [int(t[0]) for t in stims_shown["stimDuration"][:]]
    # print(picture_ids)
    # print(len(picture_ids))
    # print(max(picture_ids))
    # print(min(picture_ids))
    # print(picture_ids)
    roi_data = np.load(roi_data_name)
    response_data = np.load(response_data_name).squeeze()
    # response_data = np.where(roi_data == 0, 0, response_data)
    resolution = 1000 / 182
    # plt.imshow(response_data[0])
    # plt.title("Original Response before normalizing")
    # plt.colorbar()
    # plt.show()
    # print(response_data[0].shape)
    # print(response_data[0])
    response_data = lowhigh_normalize(frame=response_data, mask=roi_data, sig_high=100 / resolution, sig_low=1)
    images_responses = []
    # print(response_data.shape)
    # for id, tif_file in enumerate(tif_files):
    #     curr_image = Image.open(os.path.join('Images', tif_file)).convert("L")
    #     curr_image = np.array(curr_image)
    #     if curr_image.shape != (1920, 2560):
    #         print(tif_file, curr_image.shape)
    #         continue
    #     images[id] = curr_image
    # print(f"{response_data.shape} response data shape")
    #     images_respones.append([curr_image, response_data[id - 1]])
    counter = 0
    for i, id in enumerate(picture_ids):
        # print(i, id, "i, id")
        if id < 151:
            curr_picture = find_image(id - 1)
            if curr_picture is not None:
                if curr_picture.shape != (1920, 2560):
                    # print(curr_picture.shape, "Not right format")
                    continue
                images_responses.append([curr_picture, response_data[i]])
    # print(counter)
    # print(f"{len(images_responses)}, len image responses")
    images = np.zeros([147, 1920, 2560])
    responses_images = np.zeros([147, 270, 320])  # fix with mask
    for j, tuple in enumerate(images_responses):
        images[j] = tuple[0]
        responses_images[j] = tuple[1]
    # num_responses_to_plot = 10  # Number of responses you want to plot
    # fig, axes = plt.subplots(1, num_responses_to_plot, figsize=(15, 5))
    # for i in range(num_responses_to_plot):
    #     ax = axes[i]
    #     ax.imshow(response_data[i])
    #     ax.set_title(f"Response {i + 1}")
    #     ax.axis('off')  # Hide the axis
    # plt.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=.1)
    # plt.suptitle("Original Responses")
    # plt.savefig("original_responses.png")
    # plt.show()
    # plt.imshow(images[0])
    # plt.show()
    # print(len(responses_images), images.shape)
    # Remove None values
    # print(images_responses[0][0].shape, images_responses[0][1].shape)
    # for counter, tuple in enumerate(images_respones):
    #     if tuple[0] is None:
    #         # print(tuple[0].shape, tuple[1].shape)
    #         images_respones.pop(counter)
    # print(len(images_respones))
    return images_responses, images, responses_images


# images_responses, images, responses = match_pictures_with_response()
# images_responses, images, responses = match_pictures_with_response(stims_param_name="F0255/tseries_12/stimparams.dict",roi_data_name="F0255/1_roi_morphed.npy", response_data_name="F0255/avg_responses_12_13_14_15_F0255.npy")
import pickle as pk


def apply_pca(data, n_components, response_data=None, mask=None, pca_vector=None, pca_filepath="", scaler_filepath=""):
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
        plt.imshow(single_reconstructed_response)
        plt.title("Reconstructed Single Response from PCA Vector")
        plt.colorbar()
        plt.show()
    if response_data:
        # print(reconstructed_data.shape)
        # reshaped_responses = np.zeros((147, 270 * 320))
        reshaped_responses = np.zeros((588, 270 * 320))
        reshaped_responses[:, mask.flatten()] = reconstructed_data
        # reconstructed_responses = reshaped_responses.reshape(147, 270, 320)
        reconstructed_responses = reshaped_responses.reshape(588, 270, 320)
        reconstructed_responses[:, np.logical_not(mask)] = np.nan
        num_responses_to_plot = 10  # Number of responses you want to plot
        fig, axes = plt.subplots(1, num_responses_to_plot, figsize=(15, 5))
        for i in range(num_responses_to_plot):
            ax = axes[i]
            ax.imshow(reconstructed_responses[i])
            ax.set_title(f"Response {i + 1}")
            ax.axis('off')  # Hide the axis
        plt.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=.1)
        plt.suptitle("Reconstructed Responses After PCA")
        plt.savefig("reconstructed_responses.png")
        plt.show()
        pk.dump(pca, open(pca_filepath, "wb"))
        pk.dump(scaler, open(scaler_filepath, "wb"))
    else:
        reconstructed_images = reconstructed_data.reshape(147, 1920, 2560)
        # reconstructed_images = reconstructed_data.reshape(588, 1920, 2560)
        print(reconstructed_images.shape)
        plt.imshow(reconstructed_images[0])
        plt.title("Reconstructed Image(input) After PCA")
        plt.show()
        pk.dump(pca, open(pca_filepath, "wb"))
        pk.dump(scaler, open(scaler_filepath, "wb"))
    return transformed_data


# only select mask area:
# roi_mask = np.load("F0255/1_roi_morphed.npy") # !!Select the correct mask!!
# responses = np.array(responses).reshape(responses.shape[0], -1)[:, roi_mask.flatten()]
# print(responses.shape)
# response_data_pca = apply_pca(data=responses, n_components=25, response_data=True, mask=roi_mask, pca_vector=None)
# # response_data_pca = apply_pca(data=responses, n_components=50, response_data=True, mask=roi_mask, pca_vector=response_data_pca[10])
# np.save("responses_F0255_25_pca.npy", response_data_pca)
# print(images.shape)
# images = images.reshape(images.shape[0], 1920*2560)
# images_pca = apply_pca(data=images, n_components=147, response_data=False)
# np.save("images_F0255_147_pca.npy", images_pca)
# 95 mit 102 aber kann das bild nicht mehr erkennen
# 147 noch 100% variance

import torch
import numpy as np
import torch.nn as nn


def calculate_r_squared_images(images1, images2):
    r_squared_values = []

    for i in range(images1.shape[0]):
        # Flatten each pair of images
        actual = images1[i].flatten()
        predicted = images2[i].flatten()

        # Calculate mean of the actual values
        mean_actual = np.mean(actual)

        # Calculate SS_total and SS_residual
        ss_total = np.sum((actual - mean_actual) ** 2)
        ss_residual = np.sum((actual - predicted) ** 2)

        # Calculate R^2
        r_squared = 1 - (ss_residual / ss_total)
        r_squared_values.append(r_squared)

    return r_squared_values


def test_model(model_name, input_image, true_output, pca_name_response, scaler_name_responses, mask,
               n_components_responses, n_components_images, index):
    pca_reloaded_responses = pk.load(open(pca_name_response, 'rb'))
    scaler_reloaded_responses = pk.load(open(scaler_name_responses, 'rb'))
    model = PopulationCNN(input_size=n_components_images, output_size=n_components_responses)  # TODO make dynamic
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
    # print(f"{output_image.shape}, output_shape") # should be 1, 147 (pca components image input)
    true_output = torch.tensor(
        true_output.reshape(1, n_components_responses))  # second shape is number of pca components of response
    # print(f"{true_output.shape}, true output shape")  # should be 1, 147 (pca components image input)
    loss = nn.MSELoss()
    loss_test_image =  loss(output_image, true_output).item()
    # r_2 = calculate_r_squared_images(true_output, output_image)
    # print("R^2 vector:", r_2)
    print("true", true_output[0])
    print("model output", output_image[0])
    single_reconstructed = pca_reloaded_responses.inverse_transform(output_image.reshape(1, -1))
    single_reconstructed = scaler_reloaded_responses.inverse_transform(single_reconstructed)
    single_reconstructed_response = np.zeros(mask.shape)
    single_reconstructed_response[mask] = single_reconstructed.flatten()
    single_reconstructed_response[np.logical_not(mask)] = np.nan
    # plt.imshow(single_reconstructed_response)
    # plt.title(f"Model Output, loss: {torch.round(loss(output_image, true_output))}")
    # plt.colorbar()
    # plt.savefig("Model Output " + index)
    # plt.show()
    single_reconstructed2 = pca_reloaded_responses.inverse_transform(true_output.reshape(1, -1))
    single_reconstructed2 = scaler_reloaded_responses.inverse_transform(single_reconstructed2)
    single_reconstructed_response2 = np.zeros(mask.shape)
    single_reconstructed_response2[mask] = single_reconstructed2.flatten()
    single_reconstructed_response2[np.logical_not(mask)] = np.nan
    # plt.imshow(single_reconstructed_response2)
    # plt.title("True Output")
    # plt.colorbar()
    # plt.savefig("True Output " + index)
    # plt.show()
    r_2 = calculate_r_squared_images(single_reconstructed2, single_reconstructed)
    # Calculate the residual sum of squares
    true_output = true_output.detach().cpu().numpy()
    output_image = output_image.detach().cpu().numpy()
    ss_res = np.sum((true_output.flatten() - output_image.flatten()) ** 2)
    # Calculate the total sum of squares
    ss_tot = np.sum((true_output - np.mean(true_output)) ** 2)
    # Calculate R^2
    r_squared = 1 - (ss_res / ss_tot)
    print("R^2 images:", r_2[0],"R^2 pca vectors:", r_squared, "loss:", loss_test_image)

    return output_image, true_output, loss_test_image,r_2[0], r_squared


from model import PopulationCNN
# responses_pca = np.load("responses_F0255_25_pca.npy")
# images_pca = np.load("images_F0255_147_pca.npy")

# TODO clean up code

# test_model("trained_model_weights_500.pth", images_pca[10], responses_pca[10], "pca_responses.pkl", "scaler_responses.pkl", np.load("1_roi_morphed.npy"))
# for i in range(147):
#     test_model("trained_model_weights_F0255_100.pth", images_pca[i], responses_pca[i],
#                "pca_responses.pkl", "scaler_responses.pkl", np.load("F0255/1_roi_morphed.npy"))
# test_model("trained_model_weights_F0255_100.pth", images_pca[9], responses_pca[9],
#            "pca_responses.pkl", "scaler_responses.pkl", np.load("F0255/1_roi_morphed.npy"))
# test_model("trained_model_weights_F0255_100.pth", images_pca[51], responses_pca[51],
#            "pca_responses.pkl", "scaler_responses.pkl", np.load("F0255/1_roi_morphed.npy"))
# test_model("trained_model_weights_F0255_100.pth", images_pca[125], responses_pca[125],
#             "pca_responses.pkl", "scaler_responses.pkl", np.load("F0255/1_roi_morphed.npy"))

from scipy.stats import pearsonr


def calculate_image_correlations(images1, images2):
    correlations = []

    for i in range(images1.shape[0]):
        # Flatten each pair of images
        img1_flat = images1[i].flatten()
        img2_flat = images2[i].flatten()

        # Remove NaN and infinite values
        valid_mask = np.isfinite(img1_flat) & np.isfinite(img2_flat)
        img1_flat = img1_flat[valid_mask]
        img2_flat = img2_flat[valid_mask]

        # Calculate Pearson correlation coefficient
        if len(img1_flat) > 0 and len(img2_flat) > 0:
            corr, _ = pearsonr(img1_flat, img2_flat)
        else:
            corr = np.nan  # If no valid data points, set correlation to NaN
        correlations.append(corr)
    print(np.average(correlations))
    return correlations


def load_all_responses_and_avg():
    string_directory = "F0255/tseries_"
    strings_subdirectories = ["12", "13", "14", "15"]
    responses_all_trials = []
    for i in strings_subdirectories:
        os.listdir(string_directory + i)
        t_series = os.listdir(string_directory + i)
        responses = np.load(string_directory + i + '/' + t_series[0]).squeeze()
        print(responses.shape, "responses shape")
        plt.imshow(responses[20])
        plt.title("Single Response" + i)
        plt.colorbar()
        plt.show()
        responses_all_trials.append(responses)
    # for i, _ in enumerate(responses_all_trials):
    #     if i+1 < len(responses_all_trials):
    #         calculate_image_correlations(responses_all_trials[i], responses_all_trials[i+1])
    #     else:
    #         calculate_image_correlations(responses_all_trials[i], responses_all_trials[0])
    responses_all_trials = np.array(responses_all_trials)
    avg_response = np.mean(responses_all_trials, axis=0)
    print(f"Average Response Shape: {avg_response.shape}")
    plt.imshow(avg_response[20])
    plt.title("Average Response")
    plt.colorbar()
    plt.show()
    np.save("F0255/avg_responses_12_13_14_15_F0255.npy", avg_response)
    # print(responses1)


# load_all_responses_and_avg() # average all responses over trials


from model import train_save_model, train_save_model_cross
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def find_pixels_with_most_variance(images, num_of_pixels=1):
    variance = np.nanvar(images, axis=0)
    valid_mask = np.isfinite(variance)
    valid_variance = variance[valid_mask]
    max_variance_indices = np.argpartition(valid_variance, -num_of_pixels)[-num_of_pixels:]
    max_variance_pixels = [np.unravel_index(index, variance.shape) for index in np.flatnonzero(valid_mask)[max_variance_indices]]
    return max_variance_pixels, variance

def get_common_pixels_within_distance(pixels_list, distance=3):
    common_pixels = set(pixels_list[0])
    for pixels in pixels_list[1:]:
        new_common_pixels = set()
        for pixel in common_pixels:
            for other_pixel in pixels:
                if np.linalg.norm(np.array(pixel) - np.array(other_pixel)) <= distance:
                    new_common_pixels.add(pixel)
                    break
        common_pixels = new_common_pixels
    return common_pixels

def get_pixel_with_most_variance(num_of_pixels=1):
    _, _, responses_12 = match_pictures_with_response("F0255/tseries_12/stimparams.dict", "F0255/1_roi_morphed.npy", "F0255/tseries_12/response_array_1s_interval.npy")
    _, _, responses_13 = match_pictures_with_response("F0255/tseries_13/stimparams.dict", "F0255/1_roi_morphed.npy", "F0255/tseries_13/response_array_1s_interval.npy")
    _, _, responses_14 = match_pictures_with_response("F0255/tseries_14/stimparams.dict", "F0255/1_roi_morphed.npy", "F0255/tseries_14/response_array_1s_interval.npy")
    _, _, responses_15 = match_pictures_with_response("F0255/tseries_15/stimparams.dict", "F0255/1_roi_morphed.npy", "F0255/tseries_15/response_array_1s_interval.npy")

    max_var_pixels_12, variance_12 = find_pixels_with_most_variance(responses_12, num_of_pixels)
    max_var_pixels_13, variance_13 = find_pixels_with_most_variance(responses_13, num_of_pixels)
    max_var_pixels_14, variance_14 = find_pixels_with_most_variance(responses_14, num_of_pixels)
    max_var_pixels_15, variance_15 = find_pixels_with_most_variance(responses_15, num_of_pixels)

    common_pixels = get_common_pixels_within_distance([max_var_pixels_12, max_var_pixels_13, max_var_pixels_14, max_var_pixels_15], distance=10)

    # print("Pixels with most variance in trial 12:", max_var_pixels_12)
    # print("Pixels with most variance in trial 13:", max_var_pixels_13)
    # print("Pixels with most variance in trial 14:", max_var_pixels_14)
    # print("Pixels with most variance in trial 15:", max_var_pixels_15)
    print("Common pixels in all trials within 1 pixels:", len(common_pixels))

    # for i, pixel in enumerate(max_var_pixels_12):
    #     print(f"Variance for pixel {i} in trial 12: {variance_12[pixel]}")
    # for i, pixel in enumerate(max_var_pixels_13):
    #     print(f"Variance for pixel {i} in trial 13: {variance_13[pixel]}")
    # for i, pixel in enumerate(max_var_pixels_14):
    #     print(f"Variance for pixel {i} in trial 14: {variance_14[pixel]}")
    # for i, pixel in enumerate(max_var_pixels_15):
    #     print(f"Variance for pixel {i} in trial 15: {variance_15[pixel]}")


    plt.figure(figsize=(12, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(variance_12)
    plt.title("Variance Map - Trial 12")
    plt.colorbar()
    for pixel in max_var_pixels_12:
        plt.plot(pixel[1], pixel[0], 'ro')

    plt.subplot(2, 3, 2)
    plt.imshow(variance_13)
    plt.title("Variance Map - Trial 13")
    plt.colorbar()
    for pixel in max_var_pixels_13:
        plt.plot(pixel[1], pixel[0], 'ro')

    plt.subplot(2, 3, 3)
    plt.imshow(variance_14)
    plt.title("Variance Map - Trial 14")
    plt.colorbar()
    for pixel in max_var_pixels_14:
        plt.plot(pixel[1], pixel[0], 'ro')

    plt.subplot(2, 3, 4)
    plt.imshow(variance_15)
    plt.title("Variance Map - Trial 15")
    plt.colorbar()
    for pixel in max_var_pixels_15:
        plt.plot(pixel[1], pixel[0], 'ro')

    plt.subplot(2, 3, 5)
    plt.imshow(variance_15)
    plt.title("Common Pixels")
    plt.colorbar()
    for pixel in common_pixels:
        plt.plot(pixel[1], pixel[0], 'ro')

    plt.tight_layout()
    plt.show()

    return common_pixels, [responses_12, responses_13, responses_14, responses_15]

def get_pixel_value(responses, pixels):
    values = np.array([responses[:, row, col] for row, col in pixels])
    values = values.T
    print(values.shape)
    return values

def get_pixels_all_responses(responses_all_trials, pixels):
    all_values = np.zeros((147 * 4, len(common_pixels)))
    print(f"{all_values.shape=}")
    for index, response_trial in enumerate(responses_all_trials):
        curr_values = get_pixel_value(response_trial, pixels)
        if index == 0:
            index_0 = 0
        else:
            index_0 = 147 * (index)
        index_1 = 147 * (index + 1)
        print(f"{index_0=} {index_1=}")
        all_values[index_0:index_1, :] = curr_values
    print(f"{all_values.shape=}")
    return all_values
# common_pixels, responses = get_pixel_with_most_variance(num_of_pixels=225) # 101 common
# common_pixels, responses = get_pixel_with_most_variance(num_of_pixels=100) # 1 common
common_pixels, responses = get_pixel_with_most_variance(num_of_pixels=225)
print(f"{len(common_pixels)=}")
responses_pixels = get_pixels_all_responses(responses, common_pixels)

def pipeline_pixels(responses_pixels, images_pca, num_epochs, learning_rate, run_name):
    directory_path = run_name
    os.makedirs(directory_path, exist_ok=True)
    print(f"{images_pca.shape=}")
    train_save_model_cross(images_pca, responses_pixels, num_epochs, learning_rate,
                           run_name + "/model" + "_" + str(num_epochs) + "_" +
                           str(learning_rate), run_name + "/model_plot", None)

pipeline_pixels(responses_pixels, np.load("all_trials/images_pca_vectors_147.npy"), 500, 10e-3, "pixels")
# 1e-5



def pipeline(stims_param_filepath, roi_data_filepath, response_data_filepath, run_name, n_components_responses,
             n_components_images, learning_rate, num_epochs, test_indices):
    directory_path = run_name
    os.makedirs(directory_path, exist_ok=True)
    # match images with responses and apply filter + high low normalize
    # _, images, responses = match_pictures_with_response(stims_param_filepath, roi_data_filepath, response_data_filepath)
    # _, images_12, responses_12 = match_pictures_with_response("F0255/tseries_12/stimparams.dict",
    #                                                     "F0255/1_roi_morphed.npy",
    #                                                     "F0255/tseries_12/response_array_1s_interval.npy")
    # _, images_13, responses_13 = match_pictures_with_response("F0255/tseries_13/stimparams.dict",
    #                                                     "F0255/1_roi_morphed.npy",
    #                                                     "F0255/tseries_13/response_array_1s_interval.npy")
    # _, images_14, responses_14 = match_pictures_with_response("F0255/tseries_14/stimparams.dict",
    #                                                     "F0255/1_roi_morphed.npy",
    #                                                     "F0255/tseries_14/response_array_1s_interval.npy")
    # _, images_15, responses_15 = match_pictures_with_response("F0255/tseries_15/stimparams.dict",
    #                                                     "F0255/1_roi_morphed.npy",
    #                                                     "F0255/tseries_15/response_array_1s_interval.npy")
    # images = np.concatenate((images_12, images_13, images_14, images_15), axis=0)
    # responses = np.concatenate((responses_12, responses_13, responses_14, responses_15), axis=0)
    # np.save(run_name + "/images.npy", images)
    # np.save(run_name + "/responses.npy", responses)
    print(15 * "-" + "Matching Images to Responses completed" + 15 * "-")
    # apply pca to images and responses
    roi_mask = np.load(roi_data_filepath)
    # responses = np.array(responses).reshape(responses.shape[0], -1)[:, roi_mask.flatten()]
    # responses_12 = np.array(responses_12).reshape(responses_12.shape[0], -1)[:, roi_mask.flatten()]
    # responses_13 = np.array(responses_13).reshape(responses_13.shape[0], -1)[:, roi_mask.flatten()]
    # responses_14 = np.array(responses_14).reshape(responses_14.shape[0], -1)[:, roi_mask.flatten()]
    # responses_15 = np.array(responses_15).reshape(responses_15.shape[0], -1)[:, roi_mask.flatten()]
    # responses = np.concatenate((responses_12, responses_13, responses_14, responses_15), axis=0)
    # responses_pca = apply_pca(responses, n_components_responses, True, roi_mask, None,
    #                           run_name + "/" + "pca_responses_" + str(n_components_responses) + ".pkl",
    #                           run_name + "/scaler_responses.pkl")
    # which test images do I want to be shown here?
    # np.save(run_name + "/responses_pca_vectors_" + str(n_components_responses) + ".npy", responses_pca)
    # print(responses_pca)
    responses_pca = np.load(run_name + "/responses_pca_vectors_" + str(n_components_responses) + ".npy")
    print(15 * "-" + "Applied PCA to responses" + 15 * "-")
    # ADD option to load existing pca vectors with pca npy
    # images = images.reshape(images.shape[0], 1920 * 2560)
    # images_12 = images_12.reshape(images_12.shape[0], 1920 * 2560)
    # images_pca = apply_pca(images_12, n_components_images, False, roi_mask, None, run_name + "/" + "pca_images_"
    #                        + str(n_components_images) + ".pkl", run_name + "/scaler_images.pkl")
    # print("images_shape one trial", images_pca.shape)
    # images_pca = np.concatenate((images_pca, images_pca, images_pca, images_pca), axis=0)
    # print("images_pca all images", images_pca.shape)
    # ADD option to load existing pca vectors with pca npy
    # np.save(run_name + "/images_pca_vectors_" + str(n_components_images) + ".npy", images_pca)
    images_pca = np.load(run_name + "/images_pca_vectors_" + str(n_components_images) + ".npy") # load existing pca vectors
    print(15 * "-" + "Applied PCA to images" + 15 * "-")
    # train model
    # train_save_model(images_pca, responses_pca, num_epochs, learning_rate,
    #                  run_name + "/model" + "_" + str(num_epochs) + "_" +
    #                  str(learning_rate), run_name + "/model_plot", None)
    train_save_model_cross(images_pca, responses_pca, num_epochs, learning_rate,
                     run_name + "/model" + "_" + str(num_epochs) + "_" +
                     str(learning_rate), run_name + "/model_plot", None)
    print(15 * "-" + "Trained and saved the model" + 15 * "-")
    # for i in test_indices:
    #     print(f"indice {i}")
    #     test_model(run_name + "/model" + "_" + str(num_epochs) + "_" + str(learning_rate) + ".pth", images_pca[i], responses_pca[i],
    #            run_name + "/" + "pca_responses_" + str(n_components_responses) + ".pkl", run_name + "/scaler_responses.pkl",
    #            np.load(roi_data_filepath), n_components_responses, n_components_images, i)
    print(15 * "-" + "Tested the model" + 15 * "-")

# one trial
# pipeline("F0255/tseries_12/stimparams.dict", "F0255/1_roi_morphed.npy",
#          "F0255/avg_responses_12_13_14_15_F0255.npy", "one_pc_responses", 1,
#          147, 1e-3, 50,  [125,  51, 138,  19, 104,  12,  76,  31,  81,   9 , 26 , 96, 143 , 67 ,134])
# all trials !!
# with cross validation
# pipeline("F0255/tseries_12/stimparams.dict", "F0255/1_roi_morphed.npy",
   #       "F0255/avg_responses_12_13_14_15_F0255.npy", "all_trials", 1,
   #       147, 1e-5, 10,  list(range(147)))

def separate_trials(stims_param_filepath, roi_data_filepath, response_data_filepath, run_name, n_components_responses,
             n_components_images, learning_rate, num_epochs, test_indices):
    directory_path = run_name
    os.makedirs(directory_path, exist_ok=True)
    # match images with responses and apply filter + high low normalize
    _, images, responses = match_pictures_with_response(stims_param_filepath, roi_data_filepath,
                                                        response_data_filepath)
    np.save(run_name + "/images.npy", images)
    np.save(run_name + "/responses.npy", responses)
    print(15 * "-" + "Matching Images to Responses completed" + 15 * "-")
    # apply pca to images and responses
    roi_mask = np.load(roi_data_filepath)
    responses = np.array(responses).reshape(responses.shape[0], -1)[:, roi_mask.flatten()]
    responses_pca = apply_pca(responses, n_components_responses, True, roi_mask, None,
                              run_name + "/" + "pca_responses_" + str(n_components_responses) + ".pkl",
                              run_name + "/scaler_responses.pkl")
    # which test images do I want to be shown here?
    np.save(run_name + "/responses_pca_vectors_" + str(n_components_responses) + ".npy", responses_pca)
    print(15 * "-" + "Applied PCA to responses" + 15 * "-")
    # ADD option to load existing pca vectors with pca npy
    # images = images.reshape(images.shape[0], 1920 * 2560)
    # images_pca = apply_pca(images, n_components_images, False, roi_mask, None, run_name + "/" + "pca_images_"
    #                        + str(n_components_images) + ".pkl", run_name + "/scaler_images.pkl")
    # ADD option to load existing pca vectors with pca npy
    # np.save(run_name + "/images_pca_vectors_" + str(n_components_images) + ".npy", images_pca)
    images_pca = np.load(run_name + "/images_pca_vectors_" + str(n_components_images) + ".npy")
    print(15 * "-" + "Applied PCA to images" + 15 * "-")
    # train model
    model = PopulationCNN(input_size=n_components_images, output_size=n_components_responses)
    model.to("cpu")
    # model.load_state_dict(torch.load("separate_trials/model_5_0.001_trials_12_13_14_15.pth", weights_only=True))
    # model.eval()
    train_save_model(images_pca, responses_pca, num_epochs, learning_rate,
                     run_name + "/model" + "_" + str(num_epochs) + "_" +
                     str(learning_rate), run_name + "/model_plot", model)
    print(15 * "-" + "Trained and saved the model" + 15 * "-")
    for i in test_indices:
        print(f"indice {i}")
        losses_test = []
        _, _, loss, r_2, r_squared = test_model(run_name + "/model" + "_" + str(num_epochs) + "_" + str(learning_rate) + ".pth", images_pca[i],
                   responses_pca[i],
                   run_name + "/" + "pca_responses_" + str(n_components_responses) + ".pkl",
                   run_name + "/scaler_responses.pkl",
                   np.load(roi_data_filepath), n_components_responses, n_components_images, i)
        losses_test.append(loss)
    print("Average loss:", np.average(losses_test))
    print(15 * "-" + "Tested the model" + 15 * "-")

# test_indices = [125,  51, 138,  19, 104,  12,  76,  31,  81,   9 , 26 , 96, 143 , 67 ,134]
# train_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 139, 140, 141, 142, 144, 145, 146]
# separate_trials("F0255/tseries_15/stimparams.dict", "F0255/1_roi_morphed.npy",
#          "F0255/tseries_15/response_array_1s_interval.npy", "separate_trials", 1,
#          147, 1e-3, 5,  test_indices)



# TESTING
# test_indices = [125,51,138,19,104,12,76,31,81,9,26,96,143,67,134,272,198,285,166,251,159,223,178,228,156,173,243,290,214,281,419,345,432,313,398,306,370,325,375,303,320,390,437,361,428,566,492,579,460,545,453,517,472,522,450,467,537,584,508,575]
# #
#
# train_indices = [number for number in range(588) if number not in test_indices]
# images_pca = np.load("all_trials/images_pca_vectors_147.npy")
# responses_pca = np.load("all_trials/responses_pca_vectors_25.npy")
# losses_test = []
# r_2_list = []
# model_outputs = []
# true_outputs = []
# counter = 0
# for i in train_indices:
#     print(f"indice {i}")
#     curr_output, curr_ground_truth, loss, r_2, r_squared = test_model("all_trials/model_200_0.001.pth", images_pca[i],
#                responses_pca[i],
#                "all_trials" + "/" + "pca_responses_" + str(25) + ".pkl",
#                "all_trials" + "/scaler_responses.pkl",
#                np.load("F0255/1_roi_morphed.npy"), 25, 147, str(i))
#     model_outputs.append(curr_output)
#     true_outputs.append(curr_ground_truth)
#     losses_test.append(loss)
#     r_2_list.append(r_2)
# print(np.average(r_2_list))
# # print(counter)
# print(np.average(losses_test))
# print(np.min(losses_test))
# print(np.median(losses_test))
# print(np.max(losses_test))
#
# true_outputs = np.array(true_outputs)
# model_outputs = np.array(model_outputs)
# ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
# # Calculate the total sum of squares
# ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
# # Calculate R^2
# r_squared = 1 - (ss_res / ss_tot)
# print(r_squared, "INSGESAMT R^2")