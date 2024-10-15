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
    #plt.imshow(data)
    # plt.show()
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

# load_npy('response_array_1s_interval.npy')
# load_npy('stimulus.npy')
# load_dict("stimparams.dict")
load_npy("1_roi_morphed.npy")

