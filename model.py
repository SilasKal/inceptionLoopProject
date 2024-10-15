import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # Added for image display

# Core CNN Model (larger version)
class CoreCNN(nn.Module):
    def __init__(self):
        super(CoreCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),   # Increased filters from 32 to 64
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=7, padding=3),  # Increased filters from 64 to 128
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),  # Increased filters from 128 to 256
            nn.ELU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Added a new convolutional layer
            nn.ELU(),
        )

    def forward(self, x):
        return self.conv_layers(x)  # Output shape: [batch_size, 512, 64, 64]

# Modified Readout Layer to output 64x64 images
class ImageReadout(nn.Module):
    def __init__(self):
        super(ImageReadout, self).__init__()
        self.readout_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Adjusted input channels
            nn.ELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # To ensure output pixel values are between 0 and 1
        )

    def forward(self, x):
        return self.readout_layers(x)  # Output shape: [batch_size, 1, 64, 64]

# Full Model
class NeuralImageToImageModel(nn.Module):
    def __init__(self):
        super(NeuralImageToImageModel, self).__init__()
        self.core = CoreCNN()
        self.readout = ImageReadout()

    def forward(self, x):
        features = self.core(x)
        output_image = self.readout(features)
        return output_image  # Output shape: [batch_size, 1, 64, 64]

# Custom Dataset
class RandomImageDataset(Dataset):
    def __init__(self, num_images):
        super(RandomImageDataset, self).__init__()
        self.num_images = num_images
        # Generate random images with values between 0 and 1
        self.images = torch.rand(num_images, 1, 270, 320)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        input_image = self.images[idx]
        target_image = self.images[idx]  # Use the same image as target
        return input_image, target_image

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

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
    output_image_np = output_image.squeeze().cpu().numpy()

    # Plot the images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(input_image_np, cmap='gray')
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
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
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
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

    torch.save(model.state_dict(), 'trained_model_weights.pth')

    # Example prediction
    test_input_image = torch.rand(1, 1, 270, 320).to(device)
    predict_image(model, test_input_image)