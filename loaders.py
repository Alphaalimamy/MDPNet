import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import PolypDataset

device = "cuda" if torch.cuda.is_available() else "cpu"



def load_datasets():
    img_size = 256
    base_path = ""
    images_dir = f"{base_path}/images"
    masks_dir = f"{base_path}/masks"

    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor() 
    ])

    train_dataset = PolypDataset(images_dir, masks_dir, 
                                 image_transform=mask_transform, 
                                 mask_transform=mask_transform)
    
    # Split dataset
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    test_size_half = test_size // 2
    
    train_dataset, temp_dataset = random_split(train_dataset, [train_size, test_size])
    val_dataset, test_dataset = random_split(temp_dataset, [test_size_half, test_size_half])

    return train_dataset, val_dataset, test_dataset



def create_dataloaders(BATCH_SIZE, train_dataset, val_dataset, test_dataset):
    num_workers = torch.cuda.device_count() * 4 if device == "cuda" else 0
    print(f"num_workers:{num_workers}")
          
    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                num_workers=num_workers,
                                pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 num_workers=num_workers,
                                 pin_memory=False,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader