import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hydra
from loguru import logger
from omegaconf import DictConfig

from .ivus_processing import IvusProcessor
from .ivus_reshuffling import Reshuffeling

@hydra.main(config_path="C:/WorkingData/Documents/2_Coding/Python/pressure_curve_processing/", config_name="config")
def ivus_main(cfg: DictConfig):
    """
    Main function to process IVUS data from a folder structure.
    """
    # logger.info(f"Starting IVUS processing with the following configuration: {cfg}")

    input_dir = cfg.ivus_main.input_dir

    # Find all subfolders within the input directory
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    for subfolder in subfolders:
        rest_dir = os.path.join(subfolder, 'rest')
        stress_dir = os.path.join(subfolder, 'stress')

        if cfg.ivus_main.processing:
            if os.path.exists(rest_dir):
                logger.info(f"Processing rest directory: {rest_dir}")
                reshuffeling = Reshuffeling(rest_dir, plot=False)
                reshuffeling()

            if os.path.exists(stress_dir):
                logger.info(f"Processing stress directory: {stress_dir}")
                reshuffeling = Reshuffeling(stress_dir, plot=False)
                reshuffeling()

        if cfg.ivus_main.reshuffling:
            processing = IvusProcessor(rest_dir, stress_dir)
            processing.run()

if __name__ == "__main__":
    ivus_main()
