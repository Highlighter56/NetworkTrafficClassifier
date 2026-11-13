
from pathlib import Path
import glob
import shutil


def reset_state():

	print("=========== Reset Project State ===========")

	# ======== Determine Project Location ========

	# If running from inside src
	file_paths = glob.glob("../data/raw_data/*.csv")

	if len(file_paths) > 0:
		project_path = Path("..")

	# If running from project root
	elif Path("data/raw_data").is_dir():
		project_path = Path(".")

	else:
		print("---Error---")
		print("Could not locate project directories.")
		print("Run this script from src or the project root.")
		return

	# ======== Establish Directory Paths ========

	models_path = project_path / "models"
	cleaned_data_path = project_path / "data" / "cleaned_data"

	print("\nThe following data will be deleted:")
	print("- Previously trained models (.pkl files)")
	print("- Previously cleaned data (entire cleaned_data folder)")

	# ======== Confirm Reset ========

	confirmation = input("\nContinue? (y/n): ").strip().lower()

	if confirmation != "y":
		print("\nReset cancelled.")
		return

	# ======== Delete Trained Models ========

	print("\nRemoving trained models...")

	if models_path.is_dir():
		for model_file in models_path.glob("*.pkl"):
			model_file.unlink()
			print("Deleted:", model_file)
	else:
		print("Models directory does not exist.")

	# ======== Delete Cleaned Data ========

	print("\nRemoving cleaned data...")

	if cleaned_data_path.is_dir():
		shutil.rmtree(cleaned_data_path)
		print("Deleted:", cleaned_data_path)
	else:
		print("Cleaned data directory does not exist.")

	print("\n=========== Reset Complete ===========")
	print("The project will regenerate cleaned data and models")
	print("the next time main.py is executed.")


if __name__ == "__main__":
	reset_state()
