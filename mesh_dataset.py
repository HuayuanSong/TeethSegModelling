import json

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
import torch
import trimesh

class Mesh_Dataset(Dataset):
    """
    A Dataset class for loading and processing 3D mesh data along with their labels.

    Attributes:
    data_list_path (str): Path to the CSV file listing mesh file identifiers.
    num_classes (int): Number of classes for classification. Default is 15.
    patch_size (int): Size of each data patch. Default is 7000.
    """

    def __init__(self, data_list_path, num_classes=15, patch_size=7000):
        """
        Initializes the Mesh_Dataset object.
        """
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        int: Number of items in the dataset.
        """
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves a data item at the specified index.

        Args:
        idx (int): Index of the desired data item.

        Returns:
        dict: A sample containing 'cells', 'labels', 'A_S', 'A_L'.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_id = self.data_list.iloc[idx, 0]
        mesh_file = f"{file_id}.obj"
        label_file = f"{file_id}.json"

        # Load the mesh using trimesh
        mesh = trimesh.load(mesh_file, process=False)
        points = mesh.vertices
        ids = np.array(mesh.faces)
        cells = points[ids].reshape(-1, 9).astype(dtype='float32')

        # Load labels from the corresponding JSON file
        with open(label_file, 'r') as f:
            labels_data = json.load(f)
        labels = np.array(labels_data['labels']).astype('int32').reshape(-1, 1)

        # Normalize mesh data by moving it to the origin and scaling
        mean_point = points.mean(axis=0)
        points -= mean_point  # translating all points so the centroid is at the origin
        std_point = points.std(axis=0)
        for i in range(3):
            cells[:, i] = (cells[:, i] - mean_point[i]) / std_point[i]  # Normalize point 1
            cells[:, i+3] = (cells[:, i+3] - mean_point[i]) / std_point[i]  # Normalize point 2
            cells[:, i+6] = (cells[:, i+6] - mean_point[i]) / std_point[i]  # Normalize point 3

        X = cells
        Y = labels

        # Initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')

        # Select a subset of points for the current patch
        selected_idx = np.random.choice(len(X), size=self.patch_size, replace=False)
        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]

        # Calculate adjacency matrices using CUDA if available, else use CPU
        if torch.cuda.is_available():
            TX = torch.tensor(X_train[:, :3], device='cuda')
            TD = torch.cdist(TX, TX)
            D = TD.cpu().numpy()
        else:
            D = distance_matrix(X_train[:, :3], X_train[:, :3])

        # Create adjacency matrices based on distance thresholds
        S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S1[D<0.1] = 1.0
        S2[D<0.2] = 1.0
        S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, self.patch_size)))
        S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        # Convert numpy arrays to tensors
        X_train = torch.from_numpy(X_train.transpose(1, 0))
        Y_train = torch.from_numpy(Y_train.transpose(1, 0))
        S1 = torch.from_numpy(S1)
        S2 = torch.from_numpy(S2)

        # Package the data into a dictionary
        sample = {'cells': X_train, 'labels': Y_train, 'A_S': S1, 'A_L': S2}

        return sample

if __name__ == '__main__':
    # Example of how to use the dataset
    dataset = Mesh_Dataset('./train_list.csv')
    print(dataset.__getitem__(0))
