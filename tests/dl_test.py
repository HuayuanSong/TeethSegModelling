import pytest
import torch
from dl_models import STN3d, STNkd, MeshSegNet

## Tests for PointNet ##
# Test for STN3d
def test_stn3d_output_shape():
    input_channels = 3
    stn3d = STN3d(input_channels)
    x = torch.rand(1, input_channels, 10)  # Example input

    output = stn3d(x)
    assert output.shape == (1, 3, 3), "Output shape of STN3d should be (1, 3, 3)"

# Test for STNkd
def test_stnkd_output_shape():
    k = 64
    stnkd = STNkd(k)
    x = torch.rand(1, k, 10)  # Example input

    output = stnkd(x)
    assert output.shape == (1, k, k), "Output shape of STNkd should be (1, k, k)"

## Test for MeshSegNet ##
def test_meshsegnet_output_shape():
    num_classes = 15
    num_channels = 15
    model = MeshSegNet(num_classes, num_channels)
    x = torch.rand(1, num_channels, 6000)  # Example input
    a_s = torch.rand(6000, 6000)  # Example adjacency matrix
    a_l = torch.rand(6000, 6000)  # Example adjacency matrix

    output = model(x, a_s, a_l)
    assert output.shape == (1, 6000, num_classes), "Output shape of MeshSegNet should be (1, 6000, num_classes)"
