import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt



def seeding(seed):
    """ Seeding the randomness. """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_directory(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        print(f"Error: creating directory with name {path} {e}")


def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

# Function to plot training and validation metrics
def plot_metrics(train_metrics, val_metrics):
    train_losses, train_dcs, train_ious, train_precs, train_accs = train_metrics
    val_losses, val_dcs, val_ious, val_precs, val_accs = val_metrics

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 8))

    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xticks(ticks=epochs)
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.grid()
    plt.legend()

    # DICE plot
    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_dcs, label='Training DICE')
    plt.plot(epochs, val_dcs, label='Validation DICE')
    plt.xticks(ticks=epochs)
    plt.title('DICE Coefficient over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('DICE')
    # plt.grid()
    plt.legend()

    # IoU plot
    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_ious, label='Training IoU')
    plt.plot(epochs, val_ious, label='Validation IoU')
    plt.xticks(ticks=epochs)
    plt.title('IoU over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    # plt.grid()
    plt.legend()

    # Precision plot
    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_precs, label='Training Precision')
    plt.plot(epochs, val_precs, label='Validation Precision')
    plt.xticks(ticks=epochs)
    plt.title('Precision over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    # plt.grid()
    plt.legend()

    # Accuracy plot
    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xticks(ticks=epochs)
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()



def seeding(seed):
    """ Seeding the randomness. """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def visualize_results(images, outputs):
    """Visualize the input images and the predicted segmentation."""
    images = images.numpy().transpose(0, 2, 3, 1)  # Change from CHW to HWC format
    outputs = outputs.detach().numpy()  # Convert to numpy
    outputs = np.argmax(outputs, axis=1)  # Get the predicted class

    # Plot the images and outputs
    fig, axes = plt.subplots(nrows=2, ncols=len(images), figsize=(15, 6))
    for i in range(len(images)):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title('Input Image')
        axes[0, i].axis('off')

        axes[1, i].imshow(outputs[i], cmap='jet', alpha=0.5)  # Using a colormap for visualization
        axes[1, i].set_title('Predicted Segmentation')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()