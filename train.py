import torch
from tqdm import tqdm

from utils import dice_coefficient, iou_score, precision_score, accuracy_score
generator = torch.Generator().manual_seed(25)
# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss, running_dc, running_iou, running_prec, running_acc = 0, 0, 0, 0, 0

    for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        optimizer.zero_grad()
        y_pred = model(img)
        loss = criterion(y_pred, mask)

        # Compute metrics
        dc = dice_coefficient(y_pred, mask)
        iou = iou_score(y_pred, mask)
        precision = precision_score(y_pred, mask)
        accuracy = accuracy_score(y_pred, mask)

        running_loss += loss.item()
        running_dc += dc.item()
        running_iou += iou.item()
        running_prec += precision.item()
        running_acc += accuracy.item()

        loss.backward()
        optimizer.step()

    num_batches = idx + 1
    return (running_loss / num_batches,
            running_dc / num_batches,
            running_iou / num_batches,
            running_prec / num_batches,
            running_acc / num_batches, )


# Evaluation loop
def evaluate(model, val_dataloader, criterion):
    model.eval()
    running_loss, running_dc, running_iou, running_prec, running_acc = 0, 0, 0, 0, 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)

            # Compute metrics
            dc = dice_coefficient(y_pred, mask)
            iou = iou_score(y_pred, mask)
            precision = precision_score(y_pred, mask)
            accuracy = accuracy_score(y_pred, mask)

            running_loss += loss.item()
            running_dc += dc.item()
            running_iou += iou.item()
            running_prec += precision.item()
            running_acc += accuracy.item()

    num_batches = idx + 1
    return (running_loss / num_batches,
            running_dc / num_batches,
            running_iou / num_batches,
            running_prec / num_batches,
            running_acc / num_batches)

