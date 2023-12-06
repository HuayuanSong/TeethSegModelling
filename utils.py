import os
import json

import numpy as np
import visdom

class VisdomLinePlotter(object):
    """
    A class for plotting data using Visdom.

    Attributes:
    env_name (str): Name of the Visdom environment.
    """

    def __init__(self, env_name='main'):
        """
        Initialize the VisdomLinePlotter.

        Args:
        env_name (str): The environment name for Visdom.
        """
        self.viz = visdom.Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        """
        Plot or update a line in a Visdom plot.

        Args:
        var_name (str): Name of the variable to plot.
        split_name (str): Name of the split (e.g., 'train', 'val').
        title_name (str): Title of the plot.
        x (int or float): X-coordinate.
        y (int or float): Y-coordinate.
        """
        if var_name not in self.plots:
            # Create a new plot if it doesn't exist
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            # Update the existing plot
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')

def get_avail_gpu():
    """
    Detect the first available GPU on a Linux system using nvidia-smi.

    Returns:
    int: The index of the first available GPU, or 0 if none are available.
    """
    result = os.popen("nvidia-smi").readlines()
    try:
        # Find the line indicating GPU processes
        for i in range(len(result)):
            if 'Processes' in result[i]:
                process_idx = i
        # Count the number of GPUs
        num_gpu = 0
        for i in range(process_idx+1):
            if 'MiB' in result[i]:
                num_gpu += 1
        gpu_list = list(range(num_gpu))

        # Detect busy GPUs and remove them from the list
        for i in range(process_idx, len(result)):
            if result[i][22] == 'C':
                gpu_list.remove(int(result[i][5]))   
        return gpu_list[0]
    except:
        print('No GPU available, returning 0')
        return 0

def load_json(file_path: str) -> dict:
    """
    Load and return the contents of a JSON file.

    Parameters:
    file_path (str): Path to the JSON file.

    Returns:
    dict: The contents of the JSON file.
    """
    with open(file_path, "r") as f:
        json_file = json.load(f)
    return json_file
