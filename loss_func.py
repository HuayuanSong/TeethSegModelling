import numpy as np
import torch

def points_distance(points, labels, n_classes):
    """
    Calculate the maximum distance of points in each class from their mean.

    Parameters:
    points: A numpy array of points.
    labels: Corresponding labels for each point.
    n_classes: The number of unique classes.

    Returns:
    A dictionary with the class label as key and the maximum distance from the mean as value.
    """
    label_list = [n for n in range(0, n_classes)]
    tooth_dic = {}
    tooth_distance = {}

    # Loop through each class label and calculate distances.
    for i in label_list:
        key = i
        tooth_list = []
        distance = []

        # Collect points belonging to the current class label.
        for k in range(len(labels)):
            if labels[k] == i:
                tooth_list.append(points[k])

        # If the class label is present, calculate the maximum distance from mean.
        if tooth_list != []:
            mean = np.mean(tooth_list, axis=0)
            tooth_dic[key] = mean
            for point in tooth_list:
                distance.append(np.linalg.norm(point - mean, ord=2))
            distance.sort(reverse=True)
            tooth_distance[key] = distance[0]
        else:
            tooth_distance[key] = 0  # Assign 0 if the class label is missing.
    return tooth_distance

def distance_count(points, y_pre_max):
    """
    Compute the smoothness loss for points based on their distances.

    Parameters:
    points: A tensor of points.
    y_pre_max: Tensor representing the maximum predicted values.

    Returns:
    The smoothness loss for the given points.
    """
    r = 0.2
    L_smooth = 0.
    device = points.device
    one_array_points = torch.ones(points.shape).to(device)
    one_array_pred = torch.ones(y_pre_max.values.shape).to(device)

    # Calculate the smoothness loss.
    for i in range(0, points.shape[0]):
        distance_array = torch.linalg.norm((points - one_array_points * points[i]), ord=2, axis=1, keepdims=True)
        distance_sort, idx = torch.sort(distance_array)
        distance_array_pred = torch.le(distance_array, distance_sort[15])
        n = torch.count_nonzero(distance_sort[0:15])
        L_smooth += torch.linalg.norm((y_pre_max.values.reshape(distance_array_pred.shape) * distance_array_pred - y_pre_max.values[i] * one_array_pred.reshape(distance_array_pred.shape) * distance_array_pred), ord=2)/n
    return L_smooth/points.shape[0]

def smooth_Loss(y_pred, inputs):
    """
    Calculate the smoothness loss for a batch of predictions and inputs.

    Parameters:
    y_pred: Predicted values.
    inputs: Input values.

    Returns:
    The total smoothness loss for the batch.
    """
    batch_size = y_pred.shape[0]
    L_smooth = 0.

    # Accumulate the smoothness loss for each item in the batch.
    for number_patient in range(0, batch_size):
        y_pre_max = torch.max(y_pred[number_patient, :], 1)
        points = inputs[number_patient, 9:12, :].T
        L_smooth += distance_count(points, y_pre_max)
    return L_smooth

def Generalized_Dice_Loss(y_pred, y_true, class_weights, inputs, smooth=1.0):
    """
    Calculate the Generalized Dice Loss for given predictions and true values.

    Parameters:
    y_pred: Predicted probabilities [n_classes, x, y, z].
    y_true: True one-hot encoded values [n_classes, x, y, z].
    class_weights: Weights for each class.
    smooth: Smoothing parameter to avoid division by zero.

    Returns:
    The total loss combining dice loss, size constraint, and smoothness loss.
    """
    smooth = 1e-7
    Lambda_size = 0.1
    Lambda_smooth = 2.5
    weight_smooth = 1000
    size_loss = 0.
    loss = 0.
    n_classes = y_pred.shape[-1]
    batch_size = y_pred.shape[0]
    y_pre_np = y_pred.cpu().detach().numpy()
    y_tru_np = y_true.cpu().detach().numpy()

    # Calculate smoothness loss.
    L_smooth = smooth_Loss(y_pred, inputs)
    L_smooth = L_smooth * weight_smooth

    # Calculate size constraint loss.
    L_pre = []
    L_gt = []
    predicted_labels = np.zeros((y_pred.shape[1]))
    true_labels = np.zeros((y_true.shape[1]))
    for number_patient in range(0, batch_size):
        for i_label in range(n_classes):
            predicted_labels[np.argmax(y_pre_np[number_patient, :], axis=-1) == i_label] = i_label
            true_labels[np.argmax(y_tru_np[number_patient, :], axis=-1) == i_label] = i_label
        points = inputs[number_patient, 9:12, :].cpu().detach().numpy().T
        L_pre_dic = points_distance(points, predicted_labels, n_classes)
        L_gt_dic = points_distance(points, true_labels, n_classes)
        for elem in L_gt_dic:
            if elem == 0:
                continue
            else:
                L_pre.append(L_pre_dic[elem])
                L_gt.append(L_gt_dic[elem])
        size_loss += np.linalg.norm(np.array(L_gt) - np.array(L_pre), ord=2)
    size_loss = Lambda_size * size_loss

    # Calculate dice loss with weight.
    for c in range(0, n_classes):
        pred_flat = y_pred[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c] / class_weights.sum()
        loss += w * (1 - ((2. * intersection + smooth) /
                          (pred_flat.sum() + true_flat.sum() + smooth)))

    # Combine all losses.
    all_loss = (loss + size_loss + L_smooth)/2
    return all_loss
