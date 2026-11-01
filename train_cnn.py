from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from cnn import ECG_CNN
from data_loader import create_dataloaders

#setup
#train and validation
#run