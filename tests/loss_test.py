import pytest
import numpy as np
import torch
from loss_func import points_distance, distance_count, smooth_Loss, Generalized_Dice_Loss

# Sample data for testing
points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
labels = np.array([0, 1, 1])
n_classes = 2
y_pred = torch.rand(1, 10, 10, 3)
y_true = torch.rand(1, 10, 10, 3)
class_weights = torch.rand(3)
inputs = torch.rand(1, 15, 10)

def test_points_distance():
    distances = points_distance(points, labels, n_classes)
    assert isinstance(distances, dict), "Output should be a dictionary"
    assert all(isinstance(value, float) for value in distances.values()), "All distances should be float values"

def test_distance_count():
    L_smooth = distance_count(torch.tensor(points), torch.tensor(labels))
    assert isinstance(L_smooth, float), "Output should be a float representing the smoothness loss"

def test_smooth_loss():
    loss = smooth_Loss(y_pred, inputs)
    assert isinstance(loss, torch.Tensor), "Output should be a torch Tensor"
    assert loss.dim() == 0, "Output should be a scalar value"

def test_generalized_dice_loss():
    dice_loss = Generalized_Dice_Loss(y_pred, y_true, class_weights, inputs)
    assert isinstance(dice_loss, torch.Tensor), "Output should be a torch Tensor"
    assert dice_loss.dim() == 0, "Output should be a scalar value"