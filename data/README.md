Dataset lives here. Current setup uses `Released_Data_NNDL_2025/` as provided (train/test zips plus mapping files) with a derived split:

- `train.csv`, `val.csv`: stratified 90/10 by subclass, columns `image, super_class, sub_class`. Image paths are relative to this directory (e.g., `Released_Data_NNDL_2025/train_images/train_images/0.jpg`).
- `Released_Data_NNDL_2025/`: original drop containing `train_data.csv`, `superclass_mapping.csv`, `subclass_mapping.csv`, `train_images/`, `test_images/`, etc.

If you swap in a new dataset, keep the CSV schema above and update `configs/default.yaml`. Use `scripts/inspect_data.py` to sanity-check splits and label distribution before training (requires Pillow; install deps in a venv with `pip install -r requirements.txt`).
