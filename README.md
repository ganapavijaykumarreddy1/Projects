# Brain Tumour Classifier

Local brain tumour MRI classifier (training + prediction) with GUI prompts for Run button usage in VSCode.

## Project Structure
- `src/` Python scripts
- `data/` (place dataset here; not committed)
- `models/` (saved models; not committed)
- `outputs/` (predictions + images; not committed)

## Setup
```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install tensorflow pillow
```

## Train
```powershell
python .\src\imageclassifier.py
```
Optional fine-tune:
```powershell
python .\src\imageclassifier.py --fine_tune --epochs 20 --fine_tune_epochs 10
```

## Predict (Run button friendly)
```powershell
python .\src\predict_image.py
```
A file picker opens; output is shown in a dialog.

## Dataset Location
Place the dataset at:
```
./data/brain tumour dataset/Training
./data/brain tumour dataset/Testing
```

## Notes
- TensorFlow GPU support is limited on native Windows; CPU training is expected.
