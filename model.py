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
from sklearn.model_selection import train_test_split
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
def train_save_model(images, responses):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    torch.manual_seed(42)

    images = images.astype(np.float32).reshape(-1, 1, images.shape[1])
    responses = responses.astype(np.float32)
    indexes = np.arange(images.shape[0])

    images_train, images_test, responses_train, responses_test, indexes_train, indexes_test = (
        train_test_split(images, responses, indexes, test_size=0.1, random_state=42))
    print("Training indexes:", indexes_train)
    print("Test indexes:", indexes_test)

    train_dataset = CustomImageDataset(num_images=images_train.shape[0], images=images_train, targets=responses_train)
    test_dataset = CustomImageDataset(num_images=images_test.shape[0], images=images_test, targets=responses_test)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print(images_train.shape[2], responses_train.shape[1])

    model = PopulationCNN(input_size=images_train.shape[2], output_size=responses_train.shape[1])
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scaler = torch.cuda.amp.GradScaler()

    # early_stopping = EarlyStopping(patience=100, min_delta=0.001)

    num_epochs = 100
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
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

        average_loss = epoch_loss / len(train_dataloader)
        train_losses.append(average_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
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
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.savefig('loss_plot_' + str(num_epochs) + '.png')
    plt.show()

    torch.save(model.state_dict(), 'trained_model_weights_F0255_' + str(num_epochs) + '.pth')
    # print("Prediction after training:")
    # predict_image(model, images_test[0], responses_test[0])
# images = np.load("images_F0255_147_pca.npy")
# responses = np.load("responses_F0255_25_pca.npy")
# print(f"shapes: {images.shape}, {responses.shape}")
# train_save_model(images, responses)