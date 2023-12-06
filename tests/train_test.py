import pytest
import torch
from loss_func import Generalized_Dice_Loss
from torch.utils.data import DataLoader

from mesh_dataset import Mesh_Dataset
from dl_models import MeshSegNet

# Smaller datasets for testing purposes
train_list_test = './train_list_test.csv'
val_list_test = './val_list_test.csv'

@pytest.fixture
def setup_data_loaders():
    training_dataset = Mesh_Dataset(data_list_path=train_list_test, num_classes=15, patch_size=6000)
    val_dataset = Mesh_Dataset(data_list_path=val_list_test, num_classes=15, patch_size=6000)
    
    train_loader = DataLoader(dataset=training_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False, num_workers=0)

    return train_loader, val_loader

def test_model_initialization():
    model = MeshSegNet(num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5)
    assert model is not None, "Model initialization failed"

def test_loss_computation(setup_data_loaders):
    train_loader, _ = setup_data_loaders
    model = MeshSegNet(num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5)
    class_weights = torch.ones(15)

    for batched_sample in train_loader:
        inputs = batched_sample['cells']
        labels = batched_sample['labels']
        A_S = batched_sample['A_S']
        A_L = batched_sample['A_L']
        one_hot_labels = torch.nn.functional.one_hot(labels[:, 0, :], num_classes=15)

        outputs = model(inputs, A_S, A_L)
        loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)

        assert loss is not None, "Loss computation failed"
        break  # Test with only the first batch