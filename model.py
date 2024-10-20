import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Core CNN
class CoreCNN(nn.Module):
    def __init__(self):
        super(CoreCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Adjusted for 3 input channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.layers(x)  # Output shape: [batch_size, 512, 60, 80]

# Image Readout
class ImageReadout(nn.Module):
    def __init__(self):
        super(ImageReadout, self).__init__()
        self.readout_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # To ensure output pixel values are between 0 and 1
            nn.Upsample(size=(270, 320), mode='bilinear', align_corners=False)  # Upsample to 270x320
        )

    def forward(self, x):
        return self.readout_layers(x)  # Output shape: [batch_size, 1, 270, 320]

# Full Model
class NeuralImageToImageModel(nn.Module):
    def __init__(self):
        super(NeuralImageToImageModel, self).__init__()
        self.core = CoreCNN()
        self.readout = ImageReadout()

    def forward(self, x):
        features = self.core(x)
        output_image = self.readout(features)
        return output_image  # Output shape: [batch_size, 1, 270, 320]

# Custom Dataset
class RandomImageDataset(Dataset):
    def __init__(self, num_images):
        super(RandomImageDataset, self).__init__()
        self.num_images = num_images
        # Generate random images with values between 0 and 1
        self.images = torch.rand(num_images, 3, 1920, 2560)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        input_image = self.images[idx]
        target_image = torch.rand(1, 270, 320)  # Random target image
        return input_image, target_image

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs):
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

def predict_image(model, input_image):
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        output_image = model(input_image)

    # Convert tensors to numpy arrays for plotting
    input_image_np = input_image.squeeze().cpu().numpy()
    print(input_image_np.shape)
    output_image_np = output_image.squeeze().cpu().numpy()
    print(output_image_np.shape)
    # Plot the images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(input_image_np.transpose(1, 2, 0))  # Transpose to (H, W, C)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Output Image')
    plt.imshow(output_image_np, cmap='gray')
    plt.axis('off')

    plt.show()

    return output_image

# Main script
if __name__ == "__main__":
    # Check for MPS device
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create the dataset and dataloader
    dataset = RandomImageDataset(num_images=50)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    # Initialize the model
    model = NeuralImageToImageModel()
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Mixed Precision Training (if applicable)
    scaler = torch.cuda.amp.GradScaler()

    # Train the model
    num_epochs = 2
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
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

        average_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

    torch.save(model.state_dict(), 'trained_model_weights.pth')

    # Example prediction
    test_input_image = torch.rand(1, 3, 1920, 2560).to(device)
    predict_image(model, test_input_image)