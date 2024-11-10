import torch
import torch.nn as nn
import torch.optim as optim
from numpy.f2py.auxfuncs import throw_error
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


def train_save_model_cross(images, responses, num_epochs, learning_rate, model_filepath, plot_filepath, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    torch.manual_seed(42)

    images = images.astype(np.float32).reshape(-1, 1, images.shape[1])
    responses = responses.astype(np.float32)

    # Create groups to ensure the same images are not mixed between training and testing sets

    fold = 0
    all_train_losses = []
    all_val_losses = []
    all_r_2_losses = []
    # validation_indices, training_indices = generate_random_indices(num_folds, 147/num_folds, 146)
    indices = np.arange(147)
    np.random.shuffle(indices)  # Shuffle the indices
    num_folds = 10
    folds = np.array_split(indices, num_folds)
    print(folds)
    for ind, i in enumerate(folds):
        indices_2 = folds[ind] + 147
        indices_3 = indices_2 + 147
        indices_4 = indices_3 + 147
        folds[ind] = np.concatenate((i, indices_2, indices_3, indices_4))
        print(len(folds[ind]))
    # for i, indices in enumerate(random_indices_variations):
    #     print(f"Variation {i + 1}: {indices}")
    overall_outputs = []
    overall_true_outputs = []
    for index in range(num_folds):
        # print(train_index % 147, val_index % 147)
        val_index = folds[index]
        train_index = np.concatenate([folds[i] for i in range(num_folds) if i != index])
        print(train_index, val_index)
        print(len(train_index), len(val_index))
        print(len(set(train_index % 147) - set(val_index % 147)))
        fold += 1
        print(f"Fold {fold}")

        images_train, images_val = images[train_index], images[val_index]
        responses_train, responses_val = responses[train_index], responses[val_index]

        train_dataset = CustomImageDataset(num_images=images_train.shape[0], images=images_train, targets=responses_train)
        val_dataset = CustomImageDataset(num_images=images_val.shape[0], images=images_val, targets=responses_val)
        train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

        model = PopulationCNN(input_size=images_train.shape[2], output_size=responses_train.shape[1])
        model.to(device)

        # criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        val_losses = []
        r_2_losses = []

        for epoch in range(num_epochs):
            model.train()
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
            all_outputs = []
            all_true_outputs = []
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    all_outputs.append(outputs.cpu().numpy())
                    all_true_outputs.append(targets.cpu().numpy())
                    overall_outputs.append(outputs.cpu().numpy())
                    overall_true_outputs.append(targets.cpu().numpy())

            true_outputs = np.array(all_true_outputs)
            model_outputs = np.array(all_outputs)
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

    # Plot the losses and R^2 scores for each fold
    plt.figure(figsize=(10, 6))
    for i in range(num_folds):
        # plt.plot(all_train_losses[i], label=f'Training Loss Fold {i+1}')
        plt.plot(all_val_losses[i], label=f'Validation Loss Fold {i+1}')
        plt.plot(all_r_2_losses[i], label=f'R^2 Loss Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.savefig(plot_filepath + '_cross_validation.png')
    plt.show()
    for i in range(num_folds):
        print(f"Fold {i + 1} - Min Validation Loss: {min(all_val_losses[i])}, Max R^2: {max(all_r_2_losses[i])}")
    true_outputs = np.array(overall_true_outputs)
    model_outputs = np.array(overall_outputs)
    ss_res = np.sum((true_outputs.flatten() - model_outputs.flatten()) ** 2)
    ss_tot = np.sum((true_outputs - np.mean(true_outputs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Overall R^2 for cross validation: {r_squared:.4f}")
    torch.save(model.state_dict(), model_filepath + '_cross_validation.pth')

def train_save_model(images, responses, num_epochs, learning_rate, model_filepath, plot_filepath, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    torch.manual_seed(42)

    images = images.astype(np.float32).reshape(-1, 1, images.shape[1])
    responses = responses.astype(np.float32)
    indexes = np.arange(images.shape[0])

    # images_train, images_test, responses_train, responses_test, indexes_train, indexes_test = (
        # train_test_split(images, responses, indexes, test_size=0.1, random_state=42))
    indexes_test =  [125,51,138,19,104,12,76,31,81,9,26,96,143,67,134,272,198,285,166,251,159,223,178,228,156,173,243,290,214,281,419,345,432,313,398,306,370,325,375,303,320,390,437,361,428,566,492,579,460,545,453,517,472,522,450,467,537,584,508,575]
    indexes_train = [num for num in range(588) if num not in indexes_test]
    images_train, images_test = images[indexes_train], images[indexes_test]
    responses_train, responses_test = responses[indexes_train], responses[indexes_test]
    print("Training indexes:", indexes_train)
    print("Test indexes:", indexes_test)
    train_dataset = CustomImageDataset(num_images=images_train.shape[0], images=images_train, targets=responses_train)
    test_dataset = CustomImageDataset(num_images=images_test.shape[0], images=images_test, targets=responses_test)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    print(images_train.shape[2], responses_train.shape[1])
    if model is None:
        model = PopulationCNN(input_size=images_train.shape[2], output_size=responses_train.shape[1])
    model.to(device)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scaler = torch.cuda.amp.GradScaler()

    # early_stopping = EarlyStopping(patience=100, min_delta=0.001)

    train_losses = []
    val_losses = []
    r_2_losses = []
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
        all_outputs = []
        all_true_outputs = []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_outputs.append(outputs.cpu().numpy())
                all_true_outputs.append(targets.cpu().numpy())
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
    plt.plot(r_2_losses, label='R^2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.savefig(plot_filepath + '.png')
    plt.show()
    torch.save(model.state_dict(), model_filepath + '.pth')
    # print("Prediction after training:")
    # predict_image(model, images_test[0], responses_test[0])
# images = np.load("images_F0255_147_pca.npy")
# responses = np.load("responses_F0255_25_pca.npy")
# print(f"shapes: {images.shape}, {responses.shape}")
# train_save_model(images, responses)