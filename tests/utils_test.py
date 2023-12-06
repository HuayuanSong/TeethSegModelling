import pytest
import json
from utils import VisdomLinePlotter, get_avail_gpu, load_json

class TestVisdomLinePlotter:
    @pytest.fixture
    def setup_plotter(self):
        plotter = VisdomLinePlotter()
        return plotter

    def test_plot_creation(self, setup_plotter):
        plotter = setup_plotter
        var_name = "test_var"
        split_name = "train"
        title_name = "Test Plot"
        x, y = 1, 2

        plotter.plot(var_name, split_name, title_name, x, y)

        assert var_name in plotter.plots, "Plot should be created and stored in plots dictionary"

def test_gpu_availability():
    gpu_idx = get_avail_gpu()
    assert isinstance(gpu_idx, int), "GPU index should be an integer"

def test_load_json(tmpdir):
    # Create a temporary JSON file for testing
    data = {"key": "value"}
    file_path = tmpdir.join("test.json")
    with open(file_path, "w") as f:
        json.dump(data, f)

    loaded_data = load_json(file_path)
    
    assert isinstance(loaded_data, dict), "Loaded data should be a dictionary"
    assert loaded_data == data, "Loaded data should match the original data"
