import pytest
from mesh_dataset import Mesh_Dataset

class TestMeshDataset:
    @pytest.fixture
    def setup_dataset(self):
        dataset = Mesh_Dataset('./train_list.csv', num_classes=15, patch_size=7000)
        return dataset

    def test_dataset_length(self, setup_dataset):
        dataset = setup_dataset
        assert len(dataset) > 0, "Dataset length should be greater than zero"

    def test_get_item_shape(self, setup_dataset):
        dataset = setup_dataset
        sample = dataset.__getitem__(0)

        assert isinstance(sample, dict), "Output should be a dictionary"
        assert 'cells' in sample and 'labels' in sample, "Output dictionary should contain 'cells' and 'labels'"
        
        cells_shape = sample['cells'].shape
        labels_shape = sample['labels'].shape

        assert cells_shape[0] == 7000, "Cells should have 7000 rows for patch_size of 7000"
        assert labels_shape[0] == 1, "Labels should have 1 row"
        assert cells_shape[1] == labels_shape[1], "Cells and labels should have same number of columns"

    def test_adjacency_matrices(self, setup_dataset):
        dataset = setup_dataset
        sample = dataset.__getitem__(0)

        assert 'A_S' in sample and 'A_L' in sample, "Output should contain adjacency matrices 'A_S' and 'A_L'"
        
        A_S_shape = sample['A_S'].shape
        A_L_shape = sample['A_L'].shape

        assert A_S_shape == (7000, 7000), "Adjacency matrix A_S should have shape (7000, 7000)"
        assert A_L_shape == (7000, 7000), "Adjacency matrix A_L should have shape (7000, 7000)"