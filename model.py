import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
# Core CNN
import torch
import torch.nn as nn

class PopulationCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PopulationCNN, self).__init__()

        # Core network with 3 convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

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

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # Readout layers
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)

        return x


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
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
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

# Main script
def train_save_model(images, responses):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Convert images and responses to float32 and reshape images to have 1 channel
    images = images.astype(np.float32).reshape(-1, 1, images.shape[1])
    responses = responses.astype(np.float32)

    # Create the dataset and dataloader
    dataset = CustomImageDataset(num_images=images.shape[0], images=images, targets=responses)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    print(images.shape[2], responses.shape[1])
    # Initialize the model
    model = PopulationCNN(input_size=images.shape[2], output_size=responses.shape[1])
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Mixed Precision Training (if applicable)
    scaler = torch.cuda.amp.GradScaler()

    # Train the model
    num_epochs = 250
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        average_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
        # Make a prediction after training
    print("Prediction after training:")
    predict_image(model, images[0], targets[0])
    torch.save(model.state_dict(), 'trained_model_weights_' + str(num_epochs) + '.pth')
# images = np.load("images_pca.npy")
# responses = np.load("responses_no_normalize_pca.npy")
# print(f"shapes: {images.shape}, {responses.shape}")
# train_save_model(images, responses)