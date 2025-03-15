import torch

def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice

# IoU calculation
def iou_score(y_pred, y_true, epsilon=1e-6):
    y_pred = torch.sigmoid(y_pred)  # Apply sigmoid to get probabilities
    y_pred = y_pred > 0.5  # Binarize predictions
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)


# Precision calculation
def precision_score(y_pred, y_true, epsilon=1e-6):
    y_pred = torch.sigmoid(y_pred)  # Apply sigmoid to get probabilities
    y_pred = y_pred > 0.5  # Binarize predictions
    true_positive = (y_pred * y_true).sum()
    false_positive = (y_pred * (1 - y_true)).sum()
    return (true_positive + epsilon) / (true_positive + false_positive + epsilon)


# Accuracy calculation
def accuracy_score(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)  # Apply sigmoid to get probabilities
    y_pred = y_pred > 0.5  # Binarize predictions
    correct = (y_pred == y_true).float().sum()
    return correct / y_true.numel()