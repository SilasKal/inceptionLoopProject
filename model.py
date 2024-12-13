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
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, output_size)

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
        x = self.fc7(x)  # Final layer without activation for regression
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
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-8  # Vermeidet Division durch Null
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon)))
import pickle as pk
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
import torch
import torch.nn.functional as F

# with 5000 saw image that showed some features of the original image
def optimize_image(model, input_size=None, output_size=None, model_filepath='', num_steps=5000, step_size=1000):
    if model_filepath != '':
        model = ComplexNN(input_size=input_size, output_size=output_size)
        model.load_state_dict(torch.load(model_filepath, weights_only=True))
    model.eval()

    # Initialize the input image with random values
    input_image = torch.randn(1, 1, input_size, requires_grad=True,
                              device='cuda' if torch.cuda.is_available() else 'cpu')

    # Store a copy of the initial input image
    initial_input_image = input_image.clone().detach()
    reconstruct_image_from_pca("all_trials/pca_images_147.pkl", "all_trials/scaler_images.pkl",
                               input_image.detach().cpu().numpy().squeeze())

    optimizer = torch.optim.Adam([input_image], lr=step_size)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        output = model(input_image)
        print(f"{output.mean()=}")

        # Objective: maximize the output
        loss = -output.mean()  # Negative value to maximize
        print(f"{loss=}")
        # Backward pass
        loss.backward()

        # Update the input image
        optimizer.step()

        # Check if the input image has changed
        if not torch.equal(initial_input_image, input_image):
            print(f"Step {step}: Input image has changed.")
        else:
            print(f"Step {step}: Input image has not changed.")
        initial_input_image = input_image.clone().detach()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    print(f"{input_image.detach().cpu().numpy().shape=}")
    reconstruct_image_from_pca("all_trials/pca_images_147.pkl", "all_trials/scaler_images.pkl", input_image.detach().cpu().numpy().squeeze())
    return input_image.detach().cpu().numpy()

# optimize_image(None, 147, 101, 'pixels/model_250_0.01_cross_validation.pth', 1000, 0.01)
def train_save_model_cross(images, responses, num_epochs, learning_rate, model_filepath, plot_filepath, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    # torch.manual_seed(12)

    images = images.astype(np.float32).reshape(-1, 1, images.shape[1])
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
        train_dataset = CustomImageDataset(num_images=images_train.shape[0], images=images_train, targets=responses_train)
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
        # model = PopulationCNN(input_size=images_train.shape[2], output_size=responses_train.shape[1])
        model = ComplexNN(input_size=images_train.shape[2], output_size=responses_train.shape[1])
        # model = CNNtoFCNN(input_size=images_train.shape[2], output_size=responses_train.shape[1])
        model.to(device)

        # criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        criterion = MAPELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-3)
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
                if epoch == num_epochs - 1 and index == 1:
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

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
                    if epoch == num_epochs-1:
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
    for i in range(num_folds):
        plt.plot(all_train_losses[i], label=f'Training Loss Fold {i+1}')
        # plt.plot(all_val_losses[i], label=f'Validation Loss Fold {i+1}')
        # plt.plot(all_r_2_losses[i], label=f'R^2 Loss Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    # plt.ylim(y_limit)
    plt.legend()
    plt.savefig(plot_filepath + 'training_cross_validation.png')
    plt.show()
    plt.figure(figsize=(10, 6))
    for i in range(num_folds):
        # plt.plot(all_train_losses[i], label=f'Training Loss Fold {i + 1}')
        plt.plot(all_val_losses[i], label=f'Validation Loss Fold {i+1}')
        # plt.plot(all_r_2_losses[i], label=f'R^2 Loss Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    # plt.ylim(y_limit)
    plt.legend()
    plt.savefig(plot_filepath + 'validation_cross_validation.png')
    plt.show()
    plt.figure(figsize=(10, 6))
    for i in range(num_folds):
        # plt.plot(all_train_losses[i], label=f'Training Loss Fold {i + 1}')
        # plt.plot(all_val_losses[i], label=f'Validation Loss Fold {i + 1}')
        plt.plot(all_r_2_losses[i], label=f'R^2 Loss Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('R^2')
    plt.legend()
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
    print(f"{np.array(overall_targets_val).shape=}, {np.array(overall_model_outputs_val).shape=}")
    print(f"{np.array(overall_targets_val).squeeze()[:, 0].shape=}, {np.array(overall_model_outputs_val)[:, 0].shape=}")
    plt.plot(np.array(overall_targets_val).squeeze()[:, 0], np.array(overall_model_outputs_val)[:, 0], 'o', color="blue", label="Pixel 0")
    plt.plot(np.array(overall_targets_val).squeeze()[:, 1], np.array(overall_model_outputs_val)[:, 1], 'o',
             color="red", label="Pixel 1")
    plt.plot(np.array(overall_targets_val).squeeze()[:, 2], np.array(overall_model_outputs_val)[:, 2], 'o',
             color="green", label="Pixel 2")
    plt.plot(np.array(overall_targets_val).squeeze()[:, 3], np.array(overall_model_outputs_val)[:, 3], 'o',
             color="purple", label="Pixel 3")
    plt.plot(np.array(overall_targets_val).squeeze()[:, 4], np.array(overall_model_outputs_val)[:, 4], 'o',
             color="yellow", label="Pixel 4")
    plt.plot(np.array(overall_targets_val).squeeze()[:, 5], np.array(overall_model_outputs_val)[:, 5], 'o',
             color="black", label="Pixel 5")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("True vs. Predicted Validation Samples")
    plt.legend()
    plt.savefig(plot_filepath + 'scatter_pixel_cross_validation.png')
    plt.show()
    print(f"{np.array(overall_targets_train).shape=}, {np.array(overall_model_outputs_train).shape=}")
    print(f"{np.array(overall_targets_train).squeeze()[:, 0].shape=}, {np.array(overall_model_outputs_train)[:, 0].shape=}")
    for i in range(11):
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(overall_targets_train).squeeze()[:, i], np.array(overall_model_outputs_train)[:, i], 'o',
                 label=f"Pixel {i}")
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(f"True vs. Predicted Training Samples for Pixel {i}")
        plt.legend()
        # plt.savefig(plot_filepath + f'scatter_pixel_{i}_cross_training.png')
        plt.show()

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
    optimize_image(model, images_train.shape[2], responses_train.shape[1])

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
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)

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






# Example usage:
# model = YourModel(input_size, output_size)
# optimized_image = optimize_image(model, input_size)