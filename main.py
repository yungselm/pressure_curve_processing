import os
import glob
import hydra
from omegaconf import DictConfig
from loguru import logger
from pressure_processing.signal_processing import SignalProcessing
from pressure_processing.post_processing import PostProcessing


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to process pressure data from a folder structure.
    """
    input_dir = cfg.main.input_dir
    output_dir_data = cfg.main.output_dir_data
    output_dir_ivus = cfg.main.output_dir_ivus

    # Ensure output directories exist
    os.makedirs(output_dir_data, exist_ok=True)
    os.makedirs(output_dir_ivus, exist_ok=True)

    # Recursively find all CSV files matching the required naming pattern
    csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        logger.warning("No CSV files found in the input directory.")
        return

    logger.info(f"Found {len(csv_files)} CSV files to process.")

    for file_path in csv_files:
        # Process only files with required naming patterns
        if any(keyword in file_path for keyword in ["pressure_ade", "pressure_dobu", "pressure_rest"]):
            # Determine subdirectory structure for the output directory
            subdir = os.path.relpath(os.path.dirname(file_path), input_dir)  # Get relative subdir path
            output_subdir = os.path.join(output_dir_data, subdir)  # This should go inside test/processed/<subdir>
            os.makedirs(output_subdir, exist_ok=True)  # Ensure subdirectories exist

            output_path = os.path.join(output_subdir, os.path.basename(file_path))  # Save files inside the correct subdir

            try:
                processor = SignalProcessing(file_path, output_path)
                processor()
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        else:
            continue

    logger.info("Processing completed.")

    # Post-processing
    logger.info("Starting post-processing.")
    post_processor = PostProcessing(output_dir_data, output_dir_ivus)  # Ensure it's called with the correct dirs
    post_processor()

if __name__ == "__main__":
    main()

