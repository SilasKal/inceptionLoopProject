import os
import torch.optim as optim
from numpy.f2py.auxfuncs import throw_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle as pk
from PIL import Image

class ImageToResponseCNN(nn.Module):
    def __init__(self):
        super(ImageToResponseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 33 * 40, 1024)  # Adjusted for new input shape
        self.fc2 = nn.Linear(1024, 270 * 320)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        x = x.view(-1, 128 * 33 * 40)  # Adjusted for new input shape
        x = self.elu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 270, 320)
        return x


class PopulationCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PopulationCNN, self).__init__()

        # Core network with 3 convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)  # Add dropout layer with 50% probability
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(p=0.5)  # Add dropout layer with 50% probability
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(p=0.5)  # Add dropout layer with 50% probability
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)

        # Non-linear activation
        self.elu = nn.ELU()

        # Calculate the size after convolutional layers
        conv_output_size = input_size  # No pooling layers, so size remains the same

        # Readout layer to predict population activity
        self.fc1 = nn.Linear(64 * conv_output_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # Core layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout2(x)  # Apply dropout after activation

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout3(x)  # Apply dropout after activation

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # Readout layers
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)

        return x


class CNNtoFCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNtoFCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x = x.unsqueeze(1)  # Add channel dimension for 1D convs
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, nhead=4, num_layers=2):
        # Check if input_size is divisible by nhead
        if input_size % nhead != 0:
            raise ValueError("input_size must be divisible by nhead.")

        super(TransformerModel, self).__init__()
        encoder_layers = TransformerEncoderLayer(input_size, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Check the input dimensions
        if x.dim() == 4:
            x = x.squeeze(1)  # Remove the extra dimension if present

        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Aggregate output
        x = self.fc(x)
        return x


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, 64)  # First layer: input_size -> 64
        self.fc2 = nn.Linear(64, 32)  # Second layer: 64 -> 32
        self.fc3 = nn.Linear(32, output_size)  # Output layer: 32 -> output_size

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU after the first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU after the second layer
        x = self.fc3(x)  # Output layer (no activation, as it depends on the task)
        return x


class ComplexNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexNN, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, 256)  # First layer: input_size -> 256
        self.fc2 = nn.Linear(256, 128)  # Second layer: 256 -> 128
        self.fc3 = nn.Linear(128, 128)  # Third layer: 128 -> 128
        self.fc4 = nn.Linear(128, 64)  # Fourth layer: 128 -> 64
        self.fc5 = nn.Linear(64, 64)  # Fifth layer: 64 -> 64
        self.fc6 = nn.Linear(64, 32)  # Sixth layer: 64 -> 32
        self.fc7 = nn.Linear(32, output_size)  # Output layer: 32 -> output_size
        # Dropout layer
        self.dropout = nn.Dropout(0.3)  # Dropout with 30% probability

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)
        nn.init.xavier_uniform_(self.fc7.weight)

    def forward(self, x):
        # Forward pass with ReLU activations and dropout
        x = torch.sigmoid(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first layer
        x = torch.sigmoid(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after the second layer
        x = torch.sigmoid(self.fc3(x))
        x = self.dropout(x)  # Apply dropout after the third layer
        x = torch.sigmoid(self.fc4(x))
        x = self.dropout(x)  # Apply dropout after the fourth layer
        x = torch.sigmoid(self.fc5(x))
        x = self.dropout(x)  # Apply dropout after the fifth layer
        x = torch.sigmoid(self.fc6(x))
        x = self.fc7(x)  # Final output layer (no activation for regression tasks)
        return x


class DeeperNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeeperNN, self).__init__()

        # Define a deeper network architecture
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, output_size)

        # Dropout layers to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Forward pass with ReLU activations and dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,
                               padding=1)  # Output: (32, 220, 220)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                               padding=1)  # Output: (64, 220, 220)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 110, 110)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                               padding=1)  # Output: (128, 110, 110)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                               padding=1)  # Output: (256, 110, 110)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (256, 55, 55)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 55 * 55, 1024)  # Flatten and map to 1024 features
        self.fc2 = nn.Linear(1024, 37500)  # Map to output size # inh =9375 # exc = 37500

    def forward(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)  # Ensure batch size is preserved
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomCNN_2(nn.Module):
    def __init__(self):
        super(CustomCNN_2, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,
                               padding=1)  # Output: (32, 220, 220)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                               padding=1)  # Output: (64, 220, 220)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 110, 110)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                               padding=1)  # Output: (128, 110, 110)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                               padding=1)  # Output: (256, 110, 110)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (256, 55, 55)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 55 * 55, 1024)  # Flatten and map to 1024 features
        self.fc2 = nn.Linear(1024, 9375)  # Map to output size # inh =9375 # exc = 37500

    def forward(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)  # Ensure batch size is preserved
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class DeeperNN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DeeperNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 4096)  # Start with a large number of neurons
#         self.fc2 = nn.Linear(4096, 2048)
#         self.fc3 = nn.Linear(2048, 1024)
#         self.fc4 = nn.Linear(1024, 512)
#         self.fc5 = nn.Linear(512, 256)
#         self.fc6 = nn.Linear(256, 128)
#         self.fc7 = nn.Linear(128, output_size)  # Match the number of output features
#
#         # Add activation functions and dropout for regularization
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.3)  # Dropout to avoid overfitting
#
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.xavier_uniform_(self.fc3.weight)
#         nn.init.xavier_uniform_(self.fc4.weight)
#         nn.init.xavier_uniform_(self.fc5.weight)
#         nn.init.xavier_uniform_(self.fc6.weight)
#         nn.init.xavier_uniform_(self.fc7.weight)
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc3(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc4(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc5(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc6(x))
#         x = self.fc7(x)  # No activation on the output layer for regression tasks
#         return x

# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, num_images, images, targets):
        super(CustomImageDataset, self).__init__()
        self.num_images = num_images
        self.images = images
        self.targets = targets
        # Generate random images with values between 0 and 1
        # self.targets = np.expand_dims(targets, axis=1)  # Add channel dimension
        # self.images = images  # Add channel dimension

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        input_image = self.images[idx]
        target_image = self.targets[idx]
        return input_image, target_image


# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        average_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    torch.save(model.state_dict(), 'trained_model_weights.pth')


import matplotlib.pyplot as plt


def predict_image(model, input_image, true_output=None):
    model.eval()
    input_image = input_image.astype(np.float32).reshape(-1, 1, input_image.shape[1])
    input_image = torch.tensor(input_image)
    print(input_image.shape)
    with torch.no_grad():
        output_image = model(input_image)
    print(output_image.shape)
    loss = nn.MSELoss()
    print(loss(output_image, true_output))
    print(output_image)
    print(true_output)
    # Convert tensors to numpy arrays for plotting
    input_image_np = input_image.squeeze().cpu().numpy()
    output_image_np = output_image.squeeze().cpu().numpy()
    # print(input_image_np.shape)
    # print(output_image_np.shape)
    # print(true_output.shape)

    # Plot the images
    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.title('Input Image')
    # plt.imshow(input_image_np, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.title('Output Image')
    # plt.imshow(output_image_np, cmap='gray')
    # plt.axis('off')
    #
    # plt.show()

    return output_image





class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


from sklearn.model_selection import GroupKFold


def generate_random_indices(num_variations, num_indices, max_index):
    variations_test = []
    variations_train = []
    for _ in range(num_variations):
        indices = np.random.choice(max_index, num_indices, replace=False)
        indices_2 = indices + 147
        indices_3 = indices_2 + 147
        indices_4 = indices_3 + 147
        indices = np.concatenate((indices, indices_2, indices_3, indices_4))
        indices_train = [num for num in range(588) if num not in indices]
        indices_train = np.array(indices_train)
        variations_test.append(indices)
        variations_train.append(indices_train)
    return variations_test, variations_train


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-8  # Vermeidet Division durch Null
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon)))



def reconstruct_image_from_pca(pca_filepath, scaler_filepath, image_pca_vector):
    print(f"{image_pca_vector.shape=}")
    # Reconstruct the image from the PCA vector
    pca = pk.load(open(pca_filepath, 'rb'))
    scaler = pk.load(open(scaler_filepath, 'rb'))
    reconstructed_data = pca.inverse_transform(image_pca_vector.reshape(1, -1))
    reconstructed_data = scaler.inverse_transform(reconstructed_data)
    reconstructed_images = reconstructed_data.reshape(1, 1920, 2560)
    # reconstructed_images = reconstructed_data.reshape(588, 1920, 2560)
    print(f"{reconstructed_images.shape=}")
    plt.imshow(reconstructed_images[0])
    plt.title("Reconstructed Image(input) After PCA")
    plt.show()



# with 5000 saw image that showed some features of the original image
def optimize_image(model, input_size=None, model_filepath='', num_steps=5000, pca_component=None):
    if model_filepath != '':
        # model = ComplexNN(input_size=input_size, output_size=output_size)
        # model = DeeperNN(input_size=input_size, output_size=output_size)
        # model = CustomCNN()
        model.load_state_dict(torch.load(model_filepath, weights_only=True))
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Initialize the input image with random values
    # paper CNN
    input_image = torch.randn(1, 1, input_size[1], input_size[2], requires_grad=True,
                              device='cuda' if torch.cuda.is_available() else 'cpu')  # torch.Size([batch_size, 1, 220, 220])
    input_image = input_image.to(torch.float32)  # Ensure the input image is in float32 format
    if not os.path.exists(model_filepath.replace('.pth', '')):
        os.makedirs(model_filepath.replace('.pth', ''))
    plt.imshow(input_image.detach().cpu().numpy().squeeze(), cmap='gray')
    plt.savefig(f"{model_filepath.replace('.pth', '')}/initial_image.png")
    print(input_image.shape)
    # CNN
    # input_image = torch.randn(1, 1, input_size, requires_grad=True,
    #                            device='cuda' if torch.cuda.is_available() else 'cpu')
    # NN
    # input_image = torch.randn(1, input_size, requires_grad=True,
    #                          device='cuda' if torch.cuda.is_available() else 'cpu')

    # Store a copy of the initial input image
    initial_input_image = input_image.clone().detach()
    # commented out for paper data + model
    # reconstruct_image_from_pca("all_trials/pca_images_147.pkl", "all_trials/scaler_images.pkl",
    #                            input_image.detach().cpu().numpy().squeeze())

    optimizer = torch.optim.Adam([input_image], lr=0.01)

    prev_loss = float('inf')
    threshold = 1e-100  # Define your threshold here

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        output = model(input_image)
        # if pca_component == None:
        # # Objective: maximize the output
        # #     loss = -output.mean()  # Negative value to maximize
        #     loss = -output[0][0:10].mean()  # Use the first output value for optimization
        # pca_component: sollte np.array mit Shape (37500,) sein
        pc_k = torch.tensor(pca_component, dtype=torch.float32, device=output.device)
        pc_k = pc_k[:output.shape[1]]
        # loss = -torch.matmul(output[0], pc_k)
        projection = -torch.matmul(pc_k, output[0])
        loss = projection.sum()
        # Backward pass
        loss.backward()

        # Update the input image
        optimizer.step()

        # Check if the loss is still going down
        if abs(prev_loss - loss.item()) < threshold:
            print(f"Stopping early at step {step}, Loss: {loss.item()}")
            break

        prev_loss = loss.item()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            # reconstruct_image_from_pca("all_trials/pca_images_147.pkl", "all_trials/scaler_images.pkl",
            #                            input_image.detach().cpu().numpy().squeeze())
            plt.imshow(input_image.detach().cpu().numpy().squeeze(), cmap='gray')
            # if directory does not exist, create it
            plt.savefig(f"{model_filepath.replace('.pth', '')}/optimized_image_step_{step}.png")
            # plt.show()
            # input_image.detach.cpu().numpy().squeeze()
        if step == 5000:
            break
    print(f"{input_image.detach().cpu().numpy().shape=}")
    # commented out for paper data + model
    # reconstruct_image_from_pca("all_trials/pca_images_147.pkl", "all_trials/scaler_images.pkl", input_image.detach().cpu().numpy().squeeze())
    return input_image.detach().cpu().numpy()


def mixup_data(X, y, alpha=0.2):
    """
    Applies mixup to a batch of data.
    Args:
        X (torch.Tensor): Input data of shape (batch_size, ...).
        y (torch.Tensor): Labels of shape (batch_size, ...).
        alpha (float): Mixup hyperparameter controlling the interpolation strength.

    Returns:
        mixed_X (torch.Tensor): Mixed input data.
        mixed_y (torch.Tensor): Mixed labels.
        lambda_val (float): Mixup coefficient.
    """
    if alpha > 0:
        lambda_val = np.random.beta(alpha, alpha)
    else:
        lambda_val = 1.0

    batch_size = X.size(0)
    index = torch.randperm(batch_size)

    mixed_X = lambda_val * X + (1 - lambda_val) * X[index, :]
    mixed_y = lambda_val * y + (1 - lambda_val) * y[index, :]

    return mixed_X, mixed_y, lambda_val


# optimize_image(None, 147, 101, 'pixels/model_250_0.01_cross_validation.pth', 1000, 0.01)
def train_save_model_cross(images, responses, num_epochs, learning_rate, model_filepath, plot_filepath, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    # torch.manual_seed(12)
    # for CNN
    images = images.astype(np.float32).reshape(-1, 1, images.shape[1])
    # for all other models
    # images = images.astype(np.float32)
    responses = responses.astype(np.float32)
    #scaler = StandardScaler()
    # responses = scaler.fit_transform(responses)

    # x = np.arange(len(responses))  # X-axis as indices of the samples
    #
    # plt.figure(figsize=(50, 6))
    # plt.plot(x, responses, 'o', label="True Values", color="blue")
    # plt.axline((147, 0), (147, 1), color='red', linestyle='--')
    # plt.axline((294, 0), (294, 1), color='red', linestyle='--')
    # plt.axline((441, 0), (441, 1), color='red', linestyle='--')
    # plt.xlabel("Sample Index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.savefig(plot_filepath + 'scatter_distribution.png')
    # plt.show()

    # Create groups to ensure the same images are not mixed between training and testing sets
    # validation_indices, training_indices = generate_random_indices(num_folds, 147/num_folds, 146)
    indices = np.arange(147)
    np.random.shuffle(indices)  # Shuffle the indices
    num_folds = 10
    folds = np.array_split(indices, num_folds)
    print(folds)
    over_model_outputs = []
    over_targets = []
    for ind, j in enumerate(folds):
        curr_indices = folds[ind].copy()

        indices_2 = curr_indices + 147
        indices_3 = curr_indices + 294
        indices_4 = curr_indices + 441
        folds[ind] = np.concatenate((folds[ind], indices_2, indices_3, indices_4))
    for i in range(num_folds):
        for j in range(i + 1, num_folds):
            overlap = set(folds[i] % 147) & set(folds[j] % 147)
            if overlap:
                print(f"Overlap between fold {i} and fold {j}: {overlap}")
    print(folds)
    fold = 0
    # throw_error()
    # for i, indices in enumerate(random_indices_variations):
    #     print(f"Variation {i + 1}: {indices}")
    # for i in range(responses.shape[1]):
    #   plt.hist(np.array(responses)[:, i], bins=30, alpha=0.5, label=f"Pixel {i}")
    # plt.legend()
    plt.show()
    all_train_losses = []
    all_val_losses = []
    all_r_2_losses = []
    overall_model_outputs_train = []
    overall_targets_train = []
    overall_model_outputs_val = []
    overall_targets_val = []
    for index in range(num_folds):
        # print(train_index % 147, val_index % 147)
        val_index = folds[index]
        train_index = np.concatenate([folds[i] for i in range(num_folds) if i != index])
        # print("VAL", val_index)
        print(len(train_index), len(val_index))
        print(len(set(train_index % 147) - set(val_index % 147)))
        fold += 1
        print(f"Fold {fold}")
        # print("Max, Min", max(responses), min(responses))
        images_train, images_val = images[train_index], images[val_index]
        responses_train, responses_val = responses[train_index], responses[val_index]
        print(f"{responses_train.shape=}")
        print(f"{responses_train[:, 0].shape=}")
        train_dataset = CustomImageDataset(num_images=images_train.shape[0], images=images_train,
                                           targets=responses_train)
        val_dataset = CustomImageDataset(num_images=images_val.shape[0], images=images_val, targets=responses_val)
        train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

        # responses_train = responses_train.squeeze()
        # responses_val = responses_val.squeeze()
        # images_train = images_train.squeeze()
        # images_val = images_val.squeeze()
        # model = LinearRegression()
        # model = Ridge(alpha=1.0)
        # model = Lasso(alpha=0.1)
        # model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        # model = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=0)
        # model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=20, random_state=0)
        #
        # # Fit the model on the training data
        # model.fit(images_train, responses_train.ravel())  # Use ravel() to make responses 1D
        #
        # # Predict and evaluate on the training set
        # train_predictions = model.predict(images_train)
        # train_mse = mean_squared_error(responses_train, train_predictions)
        # print("Train Mean Squared Error:", train_mse)
        #
        # # Predict and evaluate on the validation set
        # val_predictions = model.predict(images_val)
        # val_mse = mean_squared_error(responses_val, val_predictions)
        # print("Validation Mean Squared Error:", val_mse)
        # ss_res = np.sum((responses_val.flatten() - val_predictions.flatten()) ** 2)
        # ss_tot = np.sum((responses_val - np.mean(responses_val)) ** 2)
        # r_squared = 1 - (ss_res / ss_tot)
        # print(f"Overall R^2 for cross validation: {r_squared:.4f}")
        # all_r_2_losses.append(r_squared)

        # Deeper NN
        model = PopulationCNN(input_size=images_train.shape[2], output_size=responses_train.shape[1])
        # model = DeeperNN(input_size=images_train.shape[1], output_size=responses_train.shape[1])
        # model = ComplexNN(input_size=images_train.shape[1], output_size=responses_train.shape[1])
        # model = CNNtoFCNN(input_size=images_train.shape[2], output_size=responses_train.shape[1])
        model.to(device)

        # criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        # criterion = MAPELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-3)
        scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        val_losses = []
        r_2_losses = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Apply mixup
                # inputs, targets, _ = mixup_data(inputs, targets, 0.7)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                if epoch == num_epochs - 1 and index == 0:
                    # print("targets", targets.detach().cpu().numpy().squeeze().shape)
                    overall_model_outputs_train.extend(outputs.detach().cpu().numpy().squeeze())
                    overall_targets_train.extend(targets.detach().cpu().numpy().squeeze())
            overall_targets_train_np = np.array(overall_targets_train)
            overall_model_outputs_train_np = np.array(overall_model_outputs_train)
            ss_res = np.sum((overall_targets_train_np.flatten() - overall_model_outputs_train_np.flatten()) ** 2)
            ss_tot = np.sum((overall_targets_train_np - np.mean(overall_targets_train_np)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"Overall R^2 for training last epoch: {r_squared:.4f}")
            average_loss = epoch_loss / len(train_dataloader)
            train_losses.append(average_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

            model.eval()
            val_loss = 0.0
            all_targets_val = []
            all_outputs_val = []
            true_values_last_epoch = []
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    all_outputs_val.append(outputs.cpu().numpy())
                    all_targets_val.append(targets.cpu().numpy())
                    if epoch == num_epochs - 1:
                        overall_model_outputs_val.append(outputs.cpu().numpy().squeeze())
                        overall_targets_val.append(targets.cpu().numpy())
                        # print(overall_targets)
                        # print(overall_model_outputs)
                        # print("targets", targets.cpu().numpy()[0][0])
                        # print("outputs", outputs.cpu().numpy()[0][0])
            true_outputs = np.array(all_targets_val)
            model_outputs = np.array(all_outputs_val)
            ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
            ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"Overall R^2 for validation images: {r_squared:.4f}")
            r_2_losses.append(r_squared)
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_r_2_losses.append(r_2_losses)
    # print(all_val_indexes)
    # Plot the losses and R^2 scores for each fold
    y_limit = (0, 20)  # Change this to your desired range

    plt.figure(figsize=(10, 6))
    average_train_losses = np.mean(all_train_losses, axis=0)
    std_train_losses = np.std(all_train_losses, axis=0)
    average_val_losses = np.mean(all_val_losses, axis=0)
    std_val_losses = np.std(all_val_losses, axis=0)
    plt.plot(average_train_losses, label='Average Training Loss')
    plt.fill_between(range(len(average_train_losses)), average_train_losses - std_train_losses,
                     average_train_losses + std_train_losses, alpha=0.2, label='Training Loss Standard Deviation')
    plt.plot(average_val_losses, label='Average Validation Loss')
    plt.fill_between(range(len(average_val_losses)), average_val_losses - std_val_losses,
                     average_val_losses + std_val_losses, alpha=0.2, label='Validation Loss Standard Deviation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    legend = plt.legend(loc='lower left', framealpha=0.8, title_fontsize=14)
    for text in legend.get_texts():
        text.set_fontsize(12)  # Change font size
    plt.savefig(plot_filepath + 'training_validation_loss.png')
    plt.show()
    plt.figure(figsize=(10, 6))
    average_r2_losses = np.mean(all_r_2_losses, axis=0)
    std_r2_losses = np.std(all_r_2_losses, axis=0)
    plt.plot(average_r2_losses, label='Average R^2 Validation')
    plt.fill_between(range(len(average_r2_losses)), average_r2_losses - std_r2_losses,
                     average_r2_losses + std_r2_losses, alpha=0.2, label='R^2 Validation Standard Deviation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('R^2', fontsize=14)
    # plt.title('Average R^2 with Standard Deviation')
    legend = plt.legend(loc="lower left", framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontsize(12)  # Change font size
    plt.savefig(plot_filepath + 'validation_r2_cross_validation.png')
    plt.show()
    for i in range(num_folds):
        print(f"Fold {i + 1} - Min Validation Loss: {min(all_val_losses[i])}, Max R^2: {max(all_r_2_losses[i])}")
    true_outputs = np.array(overall_targets_val)
    model_outputs = np.array(overall_model_outputs_val)
    ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
    ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Overall R^2 for cross validation last epoch: {r_squared:.4f}")
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.multioutput import MultiOutputRegressor
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Support Vector Regression": MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1)),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=42)),
    }

    for name, small_model in models.items():
        print(f"Training and testing {name}...")
        fold_train_mse = []
        fold_train_r2 = []
        fold_test_mse = []
        fold_test_r2 = []
        for index in range(num_folds):
            val_index = folds[index]
            train_index = np.concatenate([folds[i] for i in range(num_folds) if i != index])
            images_train, images_val = images[train_index], images[val_index]
            responses_train, responses_val = responses[train_index], responses[val_index]

            train_images_flat = images_train.reshape(images_train.shape[0], -1)
            test_images_flat = images_val.reshape(images_val.shape[0], -1)

            small_model.fit(train_images_flat, responses_train)
            train_predictions = small_model.predict(train_images_flat)
            test_predictions = small_model.predict(test_images_flat)

            train_mse = mean_squared_error(responses_train, train_predictions)
            train_r2 = r2_score(responses_train, train_predictions, multioutput='uniform_average')
            test_mse = mean_squared_error(responses_val, test_predictions)
            test_r2 = r2_score(responses_val, test_predictions, multioutput='uniform_average')

            fold_train_mse.append(train_mse)
            fold_train_r2.append(train_r2)
            fold_test_mse.append(test_mse)
            fold_test_r2.append(test_r2)

            print(
                f"{name} - Average Train MSE: {np.mean(fold_train_mse):.4f}, Average Train R^2: {np.mean(fold_train_r2):.4f}, Average Test MSE: {np.mean(fold_test_mse):.4f}, Average Test R^2: {np.mean(fold_test_r2):.4f}")

    # plot pixels
    # plt.figure(figsize=(50, 6))
    # # print(f"{np.array(overall_targets_val).shape=}, {np.array(overall_model_outputs_val).shape=}")
    # print(f"{np.array(overall_targets_val).squeeze()[:, 0].shape=}, {np.array(overall_model_outputs_val)[:, 0].shape=}")
    # plt.plot(np.array(overall_targets_val).squeeze()[:, 0], np.array(overall_model_outputs_val)[:, 0], 'o', color="blue", label="Pixel 0")
    # plt.plot(np.array(overall_targets_val).squeeze()[:, 1], np.array(overall_model_outputs_val)[:, 1], 'o',
    #          color="red", label="Pixel 1")
    # plt.plot(np.array(overall_targets_val).squeeze()[:, 2], np.array(overall_model_outputs_val)[:, 2], 'o',
    #          color="green", label="Pixel 2")
    # plt.plot(np.array(overall_targets_val).squeeze()[:, 3], np.array(overall_model_outputs_val)[:, 3], 'o',
    #          color="purple", label="Pixel 3")
    # plt.plot(np.array(overall_targets_val).squeeze()[:, 4], np.array(overall_model_outputs_val)[:, 4], 'o',
    #          color="yellow", label="Pixel 4")
    # plt.plot(np.array(overall_targets_val).squeeze()[:, 5], np.array(overall_model_outputs_val)[:, 5], 'o',
    #          color="black", label="Pixel 5")
    # plt.xlabel("True Value")
    # plt.ylabel("Predicted Value")
    # plt.title("True vs. Predicted Validation Samples")
    # plt.legend()
    # plt.savefig(plot_filepath + 'scatter_pixel_cross_validation.png')
    # plt.show()
    # print(f"{np.array(overall_targets_train).shape=}, {np.array(overall_model_outputs_train).shape=}")
    # print(f"{np.array(overall_targets_train).squeeze()[:, 0].shape=}, {np.array(overall_model_outputs_train)[:, 0].shape=}")
    # for i in range(2):
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(np.array(overall_targets_train).squeeze()[:, i], np.array(overall_model_outputs_train)[:, i], 'o',
    #              label=f"Pixel {i}")
    #     plt.xlabel("True Value")
    #     plt.ylabel("Predicted Value")
    #     plt.title(f"True vs. Predicted Training Samples for Pixel {i}")
    #     plt.legend()
    #     plt.savefig(plot_filepath + f'scatter_pixel_{i}_cross_training.png')
    #     plt.show()

    # plt.figure(figsize=(50, 6))
    # plt.plot(np.array(overall_targets_val).squeeze(),np.array(overall_model_outputs_val).squeeze(), 'o')
    # plt.xlabel("True Value")
    # plt.ylabel("Predicted Value")
    # plt.title("True vs. Predicted Validation Samples")
    # plt.legend()
    # plt.savefig(plot_filepath + 'scatter_cross_validation.png')
    # plt.show()

    # plt.figure(figsize=(50, 6))
    # plt.plot(overall_targets_train, overall_model_outputs_train, 'o')
    # # plt.plot(x, model_outputs, 'o', label="Predicted Values", color="orange"
    # plt.xlabel("True Value")
    # plt.ylabel("Predicted Value")
    # plt.title("True vs. Predicted Training Samples")
    # plt.legend()
    # plt.savefig(plot_filepath + 'scatter_cross_training.png')
    # plt.show()
    # plt.savefig(plot_filepath + 'scatter_cross_validation.png')

    torch.save(model.state_dict(), model_filepath + '_cross_validation.pth')
    # print(np.average(all_r_2_losses))
    # optimize_image(model, images_train.shape[2], responses_train.shape[1])



def train_save_model_with_sklearn(images_train, responses_train, images_test, responses_test,
                                  model_type='random_forest'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)
    elif model_type == 'svr':
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=250, max_depth=10, random_state=0)
    else:
        raise ValueError("Unsupported model type")

    # Fit the model on the training data
    model.fit(images_train, responses_train)

    # Predict and evaluate on the training set
    train_predictions = model.predict(images_train)
    train_mse = mean_squared_error(responses_train, train_predictions)
    ss_res_train = np.sum((responses_train.flatten() - train_predictions.flatten()) ** 2)
    ss_tot_train = np.sum((responses_train - np.mean(responses_train)) ** 2)
    r_squared_train = 1 - (ss_res_train / ss_tot_train)
    print(f"Train Mean Squared Error ({model_type}):", train_mse)
    print(f"Train R^2 ({model_type}):", r_squared_train)

    # Predict and evaluate on the test set
    test_predictions = model.predict(images_test)
    test_mse = mean_squared_error(responses_test, test_predictions)
    ss_res = np.sum((responses_test.flatten() - test_predictions.flatten()) ** 2)
    ss_tot = np.sum((responses_test - np.mean(responses_test)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Test Mean Squared Error ({model_type}):", test_mse)
    print(f"Test R^2 ({model_type}):", r_squared)

    return model


def train_save_model(images_train, responses_train, images_test, responses_test, num_epochs, learning_rate,
                     model_filepath, plot_filepath, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    torch.manual_seed(42)

    # images = images.astype(np.float32).reshape(-1, 1, images.shape[1])
    # responses = responses.astype(np.float32)
    # indexes = np.arange(images.shape[0])
    #
    # images_train, images_test, responses_train, responses_test, indexes_train, indexes_test = (
    #      train_test_split(images, responses, indexes, test_size=0.1, random_state=42))
    # indexes_test =  [125,51,138,19,104,12,76,31,81,9,26,96,143,67,134,272,198,285,166,251,159,223,178,228,156,173,243,290,214,281,419,345,432,313,398,306,370,325,375,303,320,390,437,361,428,566,492,579,460,545,453,517,472,522,450,467,537,584,508,575]
    # indexes_train = [num for num in range(588) if num not in indexes_test]
    # images_train, images_test = images[indexes_train], images[indexes_test]
    # responses_train, responses_test = responses[indexes_train], responses[indexes_test]
    # print("Training indexes:", indexes_train)
    # print("Test indexes:", indexes_test)
    # for i in range(responses_train.shape[1]):
    #     plt.hist(np.array(responses_train)[:, i], bins=30, alpha=0.5, label=f"Pixel {i}")
    # plt.show()
    images_test = images_test.astype(np.float32)
    images_train = images_train.astype(np.float32)
    responses_test = responses_test.astype(np.float32)
    responses_train = responses_train.astype(np.float32)
    print(f"Training shapes: {images_train.shape}, {responses_train.shape}")
    train_dataset = CustomImageDataset(num_images=images_train.shape[0], images=images_train, targets=responses_train)
    test_dataset = CustomImageDataset(num_images=images_test.shape[0], images=images_test, targets=responses_test)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)

    print(images_train.shape[1], responses_train.shape[1])
    if model is None:
        # model = DeeperNN(input_size=images_train.shape[1], output_size=responses_train.shape[1])
        # model = PopulationCNN(input_size=images_train.shape[2], output_size=responses_train.shape[1])
        model = CustomCNN()
    model.to(device)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scaler = torch.cuda.amp.GradScaler()

    # early_stopping = EarlyStopping(patience=100, min_delta=0.001)

    train_losses = []
    val_losses = []
    r_2_losses = []
    train_targets_last_epoch = []
    train_model_outputs_last_epoch = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.unsqueeze(1)  # Add batch dimension if needed
            # print(inputs.shape)
            # print(targets.shape)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            if epoch == num_epochs - 1:
                # print(outputs[0][0:20])
                # print(targets[0][0:20])
                train_targets_last_epoch.extend(targets.cpu().detach().numpy())
                train_model_outputs_last_epoch.extend(outputs.cpu().detach().numpy())
        average_loss = epoch_loss / len(train_dataloader)
        train_losses.append(average_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_true_outputs = []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.unsqueeze(1)  # Add batch dimension if needed
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_true_outputs.extend(targets.cpu().numpy())
                # print(targets.cpu().numpy().shape)
        true_outputs = np.array(all_true_outputs)
        model_outputs = np.array(all_outputs)
        ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
        ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"Overall R^2 for test images: {r_squared:.4f}")
        r_2_losses.append(r_squared)
        val_loss /= len(test_dataloader)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # early_stopping(val_loss)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    legend = plt.legend(loc="lower left", framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontsize(12)
    plt.savefig(plot_filepath + '.png')
    plt.show()
    torch.save(model.state_dict(), model_filepath + '.pth')
    plt.figure(figsize=(10, 5))
    plt.plot(r_2_losses, label='R^2 Validation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('R^2', fontsize=14)
    legend = plt.legend(loc="lower left", framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontsize(12)
    plt.savefig(plot_filepath + '_r_2' + '.png')
    plt.show()
    plt.figure(figsize=(50, 6))

    # Additional plots for predicted vs ground truth
    # plt.figure(figsize=(50, 6))
    # plt.plot(train_targets_last_epoch, train_model_outputs_last_epoch, 'o')
    # # plt.plot(x, model_outputs, 'o', label="Predicted Values", color="orange"
    # plt.xlabel("True Value")
    # plt.ylabel("Predicted Value")
    # plt.title("True vs. Predicted Training Samples")
    # plt.legend()
    # plt.savefig(plot_filepath + 'scatter_cross_training.png')
    # plt.show()

    # Calculate R^2 for the training set
    train_targets_last_epoch = np.array(train_targets_last_epoch)
    train_model_outputs_last_epoch = np.array(train_model_outputs_last_epoch)
    ss_res = np.sum((train_targets_last_epoch.flatten() - train_model_outputs_last_epoch.flatten()) ** 2)
    ss_tot = np.sum((train_targets_last_epoch - np.mean(train_targets_last_epoch)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Overall R^2 for training last epoch: {r_squared:.4f}")
    # plt.figure(figsize=(50, 6))
    # plt.plot(train_targets_last_epoch[:][0], train_model_outputs_last_epoch[:][0], 'o')
    # # plt.plot(x, model_outputs, 'o', label="Predicted Values", color="orange"
    # plt.xlabel("True Value")
    # plt.ylabel("Predicted Value")
    # plt.title("True vs. Predicted Training Samples")
    # plt.legend()
    # plt.savefig(plot_filepath + 'scatter_cross_training.png')
    # plt.show()
    # plt.savefig(plot_filepath + 'scatter_cross_validation.png')
    for i in range(5):
        plt.figure(figsize=(10, 6))
        plt.plot(train_targets_last_epoch[:][i], train_model_outputs_last_epoch[:][i], 'o', label=f"Pixel {i}")
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(f"True vs. Predicted Training Samples for Pixel {i}")
        plt.legend()
        plt.savefig(plot_filepath + f'scatter_pixel_{i}_testing.png')
        # plt.show()

    # optimize_image(model, images_train.shape[1], responses_train.shape[1], model_filepath + '.pth', 1000000000, 0.01)


def train_save_model_one_trial(images, responses, num_epochs, learning_rate, model_filepath, plot_filepath, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    torch.manual_seed(42)

    images = images.astype(np.float32)
    responses = responses.astype(np.float32)
    # indexes = np.arange(images.shape[0])
    #
    images_train, images_test, responses_train, responses_test = (
        train_test_split(images, responses, test_size=0.1, random_state=42))
    # indexes_test =  [125,51,138,19,104,12,76,31,81,9,26,96,143,67,134,272,198,285,166,251,159,223,178,228,156,173,243,290,214,281,419,345,432,313,398,306,370,325,375,303,320,390,437,361,428,566,492,579,460,545,453,517,472,522,450,467,537,584,508,575]
    # indexes_train = [num for num in range(588) if num not in indexes_test]
    # images_train, images_test = images[indexes_train], images[indexes_test]
    # responses_train, responses_test = responses[indexes_train], responses[indexes_test]
    # print("Training indexes:", indexes_train)
    # print("Test indexes:", indexes_test)
    # for i in range(responses_train.shape[1]):
    #     plt.hist(np.array(responses_train)[:, i], bins=30, alpha=0.5, label=f"Pixel {i}")
    # plt.show()
    print(f"Training shapes: {images_train.shape}, {responses_train.shape}")
    train_dataset = CustomImageDataset(num_images=images_train.shape[0], images=images_train, targets=responses_train)
    test_dataset = CustomImageDataset(num_images=images_test.shape[0], images=images_test, targets=responses_test)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)

    print(images_train.shape[1], responses_train.shape[1])
    if model is None:
        model = DeeperNN(input_size=images_train.shape[1], output_size=responses_train.shape[1])
        # model = PopulationCNN(input_size=images_train.shape[1], output_size=responses_train.shape[1])
    model.to(device)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scaler = torch.cuda.amp.GradScaler()

    # early_stopping = EarlyStopping(patience=100, min_delta=0.001)

    train_losses = []
    val_losses = []
    r_2_losses = []
    train_targets_last_epoch = []
    train_model_outputs_last_epoch = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            if epoch == num_epochs - 1:
                train_targets_last_epoch.extend(targets.cpu().detach().numpy())
                train_model_outputs_last_epoch.extend(outputs.cpu().detach().numpy())
        average_loss = epoch_loss / len(train_dataloader)
        train_losses.append(average_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_true_outputs = []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_true_outputs.extend(targets.cpu().numpy())
                # print(targets.cpu().numpy().shape)
        true_outputs = np.array(all_true_outputs)
        model_outputs = np.array(all_outputs)
        ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
        ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"Overall R^2 for test images: {r_squared:.4f}")
        r_2_losses.append(r_squared)
        val_loss /= len(test_dataloader)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # early_stopping(val_loss)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    legend = plt.legend(loc="lower left", framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontsize(12)
    plt.savefig(plot_filepath + '.png')
    plt.show()
    torch.save(model.state_dict(), model_filepath + '.pth')
    plt.figure(figsize=(10, 5))
    plt.plot(r_2_losses, label='R^2 Validation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('R^2', fontsize=14)
    legend = plt.legend(loc="lower left", framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontsize(12)
    plt.savefig(plot_filepath + '_r_2' + '.png')
    plt.show()

    # Additional plots for predicted vs ground truth
    # plt.figure(figsize=(50, 6))
    # plt.plot(train_targets_last_epoch, train_model_outputs_last_epoch, 'o')
    # # plt.plot(x, model_outputs, 'o', label="Predicted Values", color="orange"
    # plt.xlabel("True Value")
    # plt.ylabel("Predicted Value")
    # plt.title("True vs. Predicted Training Samples")
    # plt.legend()
    # plt.savefig(plot_filepath + 'scatter_cross_training.png')
    # plt.show()

    # Calculate R^2 for the training set
    train_targets_last_epoch = np.array(train_targets_last_epoch)
    train_model_outputs_last_epoch = np.array(train_model_outputs_last_epoch)
    ss_res = np.sum((train_targets_last_epoch.flatten() - train_model_outputs_last_epoch.flatten()) ** 2)
    ss_tot = np.sum((train_targets_last_epoch - np.mean(train_targets_last_epoch)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Overall R^2 for training last epoch: {r_squared:.4f}")
    # for i in range(2):
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(np.array(all_true_outputs).squeeze()[:, i], np.array(all_outputs)[:, i], 'o', label=f"Pixel {i}")
    #     plt.xlabel("True Value")
    #     plt.ylabel("Predicted Value")
    #     plt.title(f"True vs. Predicted Testing Samples for Pixel {i}")
    #     plt.legend()
    #     plt.savefig(plot_filepath + f'scatter_pixel_{i}_testing.png')
    #     # plt.show()
    # Testing smaller models
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Support Vector Regression": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    print(images_train.shape)
    print(responses_train.shape)
    train_images_flat = images_train.reshape(images_train.shape[0], -1)
    test_images_flat = images_test.reshape(images_test.shape[0], -1)

    from sklearn.multioutput import MultiOutputRegressor

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Support Vector Regression": MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1)),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=42)),
    }

    train_images_flat = images_train.reshape(images_train.shape[0], -1)
    test_images_flat = images_test.reshape(images_test.shape[0], -1)

    for name, small_model in models.items():
        print(f"Training and testing {name}...")
        small_model.fit(train_images_flat, responses_train)
        train_predictions = small_model.predict(train_images_flat)
        test_predictions = small_model.predict(test_images_flat)

        train_mse = mean_squared_error(responses_train, train_predictions)
        train_r2 = r2_score(responses_train, train_predictions, multioutput='uniform_average')
        test_mse = mean_squared_error(responses_test, test_predictions)
        test_r2 = r2_score(responses_test, test_predictions, multioutput='uniform_average')

        print(
            f"{name} - Train MSE: {train_mse:.4f}, Train R^2: {train_r2:.4f}, Test MSE: {test_mse:.4f}, Test R^2: {test_r2:.4f}")

    # optimize_image(model, images_train.shape[1], responses_train.shape[1], model_filepath + '.pth', 1000000000, 0.01)


def train_save_model_cross_full_images(images, responses, num_epochs, learning_rate, model_filepath, plot_filepath,
                                       model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')
    print(f"{images.shape=}, {responses.shape=}")
    # images = images.astype(np.float32)
    responses = responses.astype(np.float32)
    # images = images.astype(np.float32).reshape(588, 1, 1920, 2560)
    images = resize_images(images, (320, 270))
    responses = np.nan_to_num(responses, nan=0.0)  # Replace NaN values with 0.0
    images = images.astype(np.float32).reshape(-1, 1, images.shape[1], images.shape[2])
    print(f"{images.shape=}, {responses.shape=}")

    # Create groups to ensure the same images are not mixed between training and testing sets
    # validation_indices, training_indices = generate_random_indices(num_folds, 147/num_folds, 146)
    indices = np.arange(147)
    np.random.shuffle(indices)  # Shuffle the indices
    num_folds = 10
    folds = np.array_split(indices, num_folds)
    print(folds)

    for ind, j in enumerate(folds):
        curr_indices = folds[ind].copy()

        indices_2 = curr_indices + 147
        indices_3 = curr_indices + 294
        indices_4 = curr_indices + 441
        folds[ind] = np.concatenate((folds[ind], indices_2, indices_3, indices_4))
    for i in range(num_folds):
        for j in range(i + 1, num_folds):
            overlap = set(folds[i] % 147) & set(folds[j] % 147)
            if overlap:
                print(f"Overlap between fold {i} and fold {j}: {overlap}")
    print(folds)
    fold = 0

    all_train_losses = []
    all_val_losses = []
    all_r_2_losses = []
    overall_model_outputs_train = []
    overall_targets_train = []
    overall_model_outputs_val = []
    overall_targets_val = []

    for index in range(num_folds):
        val_index = folds[index]
        train_index = np.concatenate([folds[i] for i in range(num_folds) if i != index])
        fold += 1
        print(f"Fold {fold}")
        images_train, images_val = images[train_index], images[val_index]
        responses_train, responses_val = responses[train_index], responses[val_index]
        train_dataset = CustomImageDataset(num_images=images_train.shape[0], images=images_train,
                                           targets=responses_train)
        val_dataset = CustomImageDataset(num_images=images_val.shape[0], images=images_val, targets=responses_val)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        model = ImageToResponseCNN()
        model.to(device)

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-3)
        scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        val_losses = []
        r_2_losses = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / (batch_idx + 1))
                del inputs, targets
                torch.cuda.empty_cache()
            average_loss = epoch_loss / len(train_dataloader)
            train_losses.append(average_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

            model.eval()
            val_loss = 0.0
            all_val_outputs = []
            all_val_true = []
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    all_val_outputs.append(outputs.cpu().numpy())
                    all_val_true.append(targets.cpu().numpy())
                    if epoch == num_epochs - 1:
                        overall_model_outputs_val.append(outputs.cpu().numpy())
                        overall_targets_val.append(targets.cpu().numpy())
                    del inputs, targets
                    torch.cuda.empty_cache()
            average_val_loss = val_loss / len(val_dataloader)
            val_losses.append(average_val_loss)
            print(f"Validation Loss: {average_val_loss:.4f}")

            true_outputs = np.array(all_val_true)
            model_outputs = np.array(all_val_outputs)
            ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
            ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            r_2_losses.append(r_squared)
            print(f"Overall R^2 for validation images: {r_squared:.4f}")

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_r_2_losses.append(r_2_losses)

    plt.figure(figsize=(10, 6))
    average_train_losses = np.mean(all_train_losses, axis=0)
    std_train_losses = np.std(all_train_losses, axis=0)
    average_val_losses = np.mean(all_val_losses, axis=0)
    std_val_losses = np.std(all_val_losses, axis=0)
    plt.plot(average_train_losses, label='Average Training Loss')
    plt.fill_between(range(len(average_train_losses)), average_train_losses - std_train_losses,
                     average_train_losses + std_train_losses, alpha=0.2, label='Training Loss Standard Deviation')
    plt.plot(average_val_losses, label='Average Validation Loss')
    plt.fill_between(range(len(average_val_losses)), average_val_losses - std_val_losses,
                     average_val_losses + std_val_losses, alpha=0.2, label='Validation Loss Standard Deviation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.title('Training and Validation Loss per Epoch')
    legend = plt.legend(loc='lower left', framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontsize(12)
    plt.savefig(plot_filepath + 'training_validation_loss.png')
    plt.show()
    plt.figure(figsize=(10, 6))
    average_r2_losses = np.mean(all_r_2_losses, axis=0)
    std_r2_losses = np.std(all_r_2_losses, axis=0)
    plt.plot(average_r2_losses, label='Average R^2 Validation')
    plt.fill_between(range(len(average_r2_losses)), average_r2_losses - std_r2_losses,
                     average_r2_losses + std_r2_losses, alpha=0.2, label='R^2 Validation Standard Deviation')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('R^2', fontsize=14)
    # plt.title('R^2')
    legend = plt.legend(loc='lower left', framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontsize(12)
    plt.savefig(plot_filepath + 'validation_r2_cross_validation.png')
    plt.show()
    for i in range(num_folds):
        print(f"Fold {i + 1} - Min Validation Loss: {min(all_val_losses[i])}, Max R^2: {max(all_r_2_losses[i])}")
    true_outputs = np.array(overall_targets_val)
    model_outputs = np.array(overall_model_outputs_val)
    ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
    ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Overall R^2 for cross validation last epoch: {r_squared:.4f}")
    # plot pixels
    plt.figure(figsize=(50, 6))
    # print(f"{np.array(overall_targets_val).shape=}, {np.array(overall_model_outputs_val).shape=}")
    print(f"{np.array(overall_targets_val).squeeze()[:, 0].shape=}, {np.array(overall_model_outputs_val)[:, 0].shape=}")
    num_samples_to_plot = 5  # Number of samples to plot
    plt.figure(figsize=(15, 10))
    print(f"{true_outputs[0].shape=}, {model_outputs[0].shape=}")
    # for i in range(num_samples_to_plot):
    #     plt.subplot(2, num_samples_to_plot, i + 1)
    #     plt.imshow(true_outputs[i].squeeze())
    #     plt.title(f'True Sample Validation {i + 1}')
    #
    #     plt.subplot(2, num_samples_to_plot, i + 1 + num_samples_to_plot)
    #     plt.imshow(model_outputs[i].squeeze())
    #     plt.title(f'Predicted Sample Validation {i + 1}')
    # plt.tight_layout()
    # plt.show()
    # num_samples_to_plot = 5  # Number of samples to plot
    # plt.figure(figsize=(15, 10))
    # for i in range(num_samples_to_plot):
    #     plt.subplot(2, num_samples_to_plot, i + 1)
    #     plt.imshow(overall_targets_train[-(i + 1)].squeeze())
    #     plt.title(f'True Sample Training {i + 1}')
    #
    #     plt.subplot(2, num_samples_to_plot, i + 1 + num_samples_to_plot)
    #     plt.imshow(overall_model_outputs_train[-(i + 1)].squeeze())
    #     plt.title(f'Predicted Sample Training {i + 1}')
    # plt.tight_layout()
    # plt.show()
    torch.save(model.state_dict(), model_filepath + '_cross_validation.pth')
    # print(np.average(all_r_2_losses))
    # optimize_image(model, images_train.shape[1], responses_train.shape[1])


def resize_images(images, new_size):
    resized_images = []
    for img in images:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(new_size)
        resized_images.append(np.array(pil_img))
    return np.array(resized_images)


# Example usage:

def train_save_model_full_images(images, responses, num_epochs, learning_rate, model_filepath, plot_filepath,
                                 model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')
    print(f"{images.shape=}, {responses.shape=}")
    images = resize_images(images, (320, 270))
    # images = images.astype(np.float32)
    responses = responses.astype(np.float32)
    # plt.imshow(images[0])
    # plt.show()
    responses = np.nan_to_num(responses, nan=0.0)  # Replace NaN values with 0.0
    images = images.astype(np.float32).reshape(-1, 1, images.shape[1], images.shape[2])
    print(f"{images.shape=}, {responses.shape=}")

    images_train, images_val, responses_train, responses_val = (
        train_test_split(images, responses, test_size=0.1, random_state=42))
    train_dataset = CustomImageDataset(num_images=images_train.shape[0], images=images_train,
                                       targets=responses_train)
    val_dataset = CustomImageDataset(num_images=images_val.shape[0], images=images_val, targets=responses_val)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Deeper NN
    model = ImageToResponseCNN()

    model.to(device)

    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    #  criterion = MAPELoss()
    # citerion = nn.MAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-3)
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []
    r_2_losses = []
    overall_model_outputs_train = []
    overall_targets_train = []
    overall_model_outputs_val = []
    overall_targets_val = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=epoch_loss / (batch_idx + 1))
            overall_targets_train.extend(targets.cpu().detach().numpy())
            overall_model_outputs_train.extend(outputs.cpu().detach().numpy())
            # del inputs, targets
            # torch.cuda.empty_cache()
            print(loss.item())
        average_loss = epoch_loss / len(train_dataloader)
        train_losses.append(average_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_true_outputs = []
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_outputs.append(outputs.cpu().numpy())
                all_true_outputs.append(targets.cpu().numpy())
                del inputs, targets
                torch.cuda.empty_cache()
        average_val_loss = val_loss / len(val_dataloader)
        val_losses.append(average_val_loss)
        print(f"Validation Loss: {average_val_loss:.4f}")

        true_outputs = np.array(all_true_outputs)
        model_outputs = np.array(all_outputs)
        ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
        ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        r_2_losses.append(r_squared)
        print(f"Overall R^2 for validation images: {r_squared:.4f}")

        if epoch == num_epochs - 1:
            overall_model_outputs_val.extend(model_outputs)
            overall_targets_val.extend(true_outputs)
    true_outputs_temp = np.array(overall_targets_train)
    model_outputs_temp = np.array(overall_model_outputs_train)
    ss_res = np.sum((true_outputs_temp.flatten() - model_outputs_temp.flatten()) ** 2)
    ss_tot = np.sum((true_outputs_temp - np.mean(true_outputs_temp)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Overall R^2 for training last epoch: {r_squared:.4f}")
    # Plot the losses and R^2 scores
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss per Epoch')
    legend = plt.legend(loc='upper right', framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontsize(12)
    plt.savefig(plot_filepath + 'training_validation_loss.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(r_2_losses, label='R^2')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('R^2', fontsize=14)
    plt.title('R^2 per Epoch', fontsize=14)
    legend = plt.legend(loc='lower right', framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontsize(12)
    plt.savefig(plot_filepath + 'r2_per_epoch.png')
    plt.show()

    print(f"Min Validation Loss: {min(val_losses)}, Max R^2: {max(r_2_losses)}")

    true_outputs = np.array(overall_targets_val)
    model_outputs = np.array(overall_model_outputs_val)
    ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
    ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Overall R^2 for cross validation last epoch: {r_squared:.4f}")
    roi_mask = np.load('F0255/1_roi_morphed.npy')
    num_samples_to_plot = 5  # Number of samples to plot
    plt.figure(figsize=(15, 10))
    true_outputs = np.where(roi_mask, true_outputs, np.nan)
    model_outputs = np.where(roi_mask, model_outputs, np.nan)
    for i in range(num_samples_to_plot):
        plt.subplot(2, num_samples_to_plot, i + 1)
        plt.imshow(true_outputs[i].squeeze())
        plt.title(f'True Sample Validation {i + 1}')

        plt.subplot(2, num_samples_to_plot, i + 1 + num_samples_to_plot)
        plt.imshow(model_outputs[i].squeeze())
        plt.title(f'Predicted Sample Validation {i + 1}')
    plt.tight_layout()
    plt.show()
    num_samples_to_plot = 5  # Number of samples to plot
    overall_targets_train = np.where(roi_mask, overall_targets_train, np.nan)
    overall_model_outputs_train = np.where(roi_mask, overall_model_outputs_train, np.nan)
    for i in range(num_samples_to_plot):
        plt.subplot(2, num_samples_to_plot, i + 1)
        plt.imshow(overall_targets_train[-(i + 1)].squeeze())
        plt.title(f'True Sample Training {i + 1}')

        plt.subplot(2, num_samples_to_plot, i + 1 + num_samples_to_plot)
        plt.imshow(overall_model_outputs_train[-(i + 1)].squeeze())
        plt.title(f'Predicted Sample Training {i + 1}')
    plt.tight_layout()
    plt.show()
    torch.save(model.state_dict(), model_filepath + '_cross_validation.pth')
    # print(np.average(all_r_2_losses))
    # optimize_image(model, images_train.shape[1], responses_train.shape[1])
