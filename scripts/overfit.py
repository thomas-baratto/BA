import torch
import torch.nn as nn
import optuna
import logging
import os
import json
import argparse
import time
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from core.trainer import main_train, evaluate
from core.data_loader import load_data, CSVDataset
from core.utils import compute_regression_metrics, ResourceLogger
from scripts.run_optuna import detect_columns_from_csv

# Add parent directory to path to import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import NeuralNetwork
from core.dataset import CSVDataset, load_data
from core
from sklearn.preprocessing import MinMaxScaler

import os
import math

if __name__ == "__main__":



