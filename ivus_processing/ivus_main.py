import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hydra
import loguru as logger
from omegaconf import DictConfig

from ivus_processing.ivus_processing import IvusProcessor
from ivus_processing.ivus_reshuffling import Reshuffeling

@hydra.ivus_main(config_path=".", config_name="config")
def ivus_main(cfg: DictConfig):
    """
    Main function to process IVUS data from a folder structure.
    """
    