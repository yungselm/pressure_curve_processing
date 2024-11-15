import os
import glob
import hydra
from omegaconf import DictConfig
from loguru import logger
from signal_processing import SignalProcessing

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to process pressure data from a folder structure.
    """
    input_dir = cfg.main.input_dir
    output_dir_data = cfg.main.output_dir_data
    output_dir_ivus = cfg.main.output_dir_ivus

    logger.info(f"Starting processing with input directory: {input_dir}")
    logger.info(f"Output directory for processed data: {output_dir_data}")
    logger.info(f"Output directory for IVUS data: {output_dir_ivus}")

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
            subdir = os.path.relpath(os.path.dirname(file_path), input_dir)
            output_path = os.path.join(output_dir_data, subdir, os.path.basename(file_path))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure subdirectories exist

            try:
                logger.info(f"Processing file: {file_path}")
                processor = SignalProcessing(file_path, output_path)
                processor()
                logger.info(f"Successfully processed and saved to: {output_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        else:
            logger.info(f"Skipped file (pattern mismatch): {file_path}")

    logger.info("Processing completed.")

if __name__ == "__main__":
    main()
