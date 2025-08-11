import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


def plot_difference_heatmap():
    def load_gray(path):
        img = mpimg.imread(path)
        if img.ndim == 3:
            img = img[..., :3]
            return np.mean(img, axis=-1)
        return img

    # Zwei Bilder laden
    img1 = load_gray('paper_data_model_10000_V1_Exc_L23_PC1/optimized_image_step_900.png')
    img2 = load_gray('paper_data_model_10000_V1_Exc_L23_PC2/optimized_image_step_900.png')

    # Gleiche Größe
    min_shape = np.min([img1.shape, img2.shape], axis=0)
    img1 = img1[:min_shape[0], :min_shape[1]]
    img2 = img2[:min_shape[0], :min_shape[1]]

    # Normalisieren (auf [0,1])
    img1 = img1 / np.max(img1)
    img2 = img2 / np.max(img2)

    # Differenz berechnen
    diff = img1 - img2

    # Plot als Heatmap
    plt.figure(figsize=(6, 5))
    im = plt.imshow(diff, cmap='seismic', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    plt.colorbar(im, label='PC1 - PC2')
    plt.axis('off')
    plt.title("Differenzheatmap: PC1 - PC2")
    plt.tight_layout()
    plt.show()


plot_difference_heatmap()
