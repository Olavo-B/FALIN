# /*****************************************************************************/
#  * File: main.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-05-20
#  * License: MIT License
#  * Description: Main file for running experiments on UCR datasets using the 
#  * Temporal2Image approach and various models.
# /*****************************************************************************/

import os
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Ensure torchinfo is imported at the top of your file
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False

from temporal2img import Temporal2ImageDataset
from trainer import run_experiment



# ---- Global Constants and Configurations ----
# (You can adjust these as needed, or move them to a config file)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

REDUNDANCE = 5 # Number of times to repeat each dataset (for testing purposes, set to 1 for no redundancy)

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MODELS_TO_RUN = ['FalinCNN', 'Falin_XGB', 'MobileNetV2', 'MobileNet_XGB']  # Options: 'MobileNetV2', 'EfficientNet', 'MobileNet_XGB'
IMG_STRATEGY = 'stft'  # Options: 'gaf', 'stft'

# XGBoost Hyperparameters (can be tuned for better performance)
N_ESTIMATORS = 50
MAX_DEPTH = 6
LEARNING_RATE_XGB = 0.1
DESCRIPTOR_SIZE = 64

DATASET = [
#   'Beef',
#   'BeetleFly',
#   'Ham',
#   'SwedishLeaf',
#   'FordA',
#   'FordB',
#   'Haptics',
#   'Symbols',
#   'Yoga',
#   'OliveOil',
#   'ECGFiveDays',
#   'HandOutlines',
#   'Adiac',
#   'Fish',
#   'Phoneme',
#   'Strawberry',
#   'Computers',
#   'Wine',
#   'StarLightCurves',
#   'FaceAll',
#   'OSULeaf',
#   'MoteStrain',
#   'Plane',
#   'Wafer',
#   'CBF',


# Intersection of the 10 biggest datasets between LSTM-FCN paper and UCR Archive (for better comparison with baselines):
'FacesUCR',
'Mallat',
'Yoga',
'FordB',
'FordA',
'TwoPatterns',
'ECG5000',
'Wafer',
'StarLightCurves',
'ElectricDevices',


]
 # Change this to the dataset you want to run (e.g., 'InlineSkate', 'ElectricDevices', etc.)


# -------------------------------------------


# Wrapper of Temporal2ImageDataset to work with UCR's (X, y) format and be compatible with DataLoader
class UCRDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, X, y, strategy='gaf'):
        self.X_dataset = Temporal2ImageDataset(X, estrategy=strategy)
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_dataset[idx], self.y[idx]

def load_and_encode_ucr(train_path, test_path):
    """Load UCR datasets and normalize labels to integers using LabelEncoder."""
    train_data = np.loadtxt(train_path, delimiter='\t')
    test_data = np.loadtxt(test_path, delimiter='\t')
    
    
    X_train, y_train_raw = train_data[:, 1:], train_data[:, 0]
    X_test, y_test_raw = test_data[:, 1:], test_data[:, 0]
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    # Sanity check for NaN values in the raw dataset (before any transformations)
    # For datasets with Variable Length, we might have NaNs due to padding. We will handle this in the Temporal2ImageDataset class.
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        print("[WARNING] NaN values detected in the raw dataset. These will be handled during the temporal to image transformation.")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
    
    return X_train, y_train, X_test, y_test, len(le.classes_), le.classes_

def save_confusion_matrix(y_true, y_pred, classes, dataset_name, model_name, save_dir):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusão - {model_name} ({dataset_name})')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'CM_{model_name}.png'))
    plt.close()

def save_hardware_complexity(models_dict, seq_length, save_dir, device):
    """
    Calculate and save MACs and Params using torchinfo.
    It writes a summary table at the top of the file, followed by 
    the detailed architectural report for each model.
    """
    if not TORCHINFO_AVAILABLE:
        print("Warning: torchinfo is not installed. Hardware metrics will not be saved.")
        return
        
    filepath = os.path.join(save_dir, 'hardware_complexity_MACs.txt')
    dummy_input = torch.randn(1, 1, seq_length, seq_length).to(device)
    
    # List to store the detailed string reports for later
    detailed_reports = []
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # 1. Write the Summary Table Header
        f.write("=== FPGA Hardware Metrics Summary ===\n")
        f.write(f"Input size: 1x1x{seq_length}x{seq_length}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Model':<20} | {'MACs (Mega)':<15} | {'Params (Mega)':<15}\n")
        f.write("-" * 65 + "\n")
        
        # 2. Iterate through models to populate the table and gather detailed reports
        for model_name, data in models_dict.items():
            model = data['model']
            
            # Select the target model to evaluate
            if model_name in ['MobileNet_XGB', 'Falin_XGB']:
                target_model = model.cnn_extractor
            else:
                target_model = model
                
            # Generate statistics with torchinfo
            # verbose=0 prevents the summary from printing to the console
            model_stats = summary(target_model, 
                                  input_data=dummy_input,
                                  col_names=['input_size', 'output_size', 'num_params', 'mult_adds'],
                                  verbose=0)
            
            # Extract MACs and Params for the summary table
            macs = model_stats.total_mult_adds
            params = model_stats.total_params
                
            # Write row to the summary table
            f.write(f"{model_name:<20} | {macs / 1e6:>10.2f} M   | {params / 1e6:>10.2f} M\n")
            
            # Save the full detailed report string to append later
            detailed_reports.append((model_name, str(model_stats)))
            
        # 3. Write the Detailed Reports section
        f.write("\n\n" + "=" * 65 + "\n")
        f.write("=== Detailed Architecture Reports ===\n")
        f.write("=" * 65 + "\n")
        
        for model_name, report in detailed_reports:
            f.write(f"\n\n>>> Detailed Report for: {model_name} <<<\n")
            f.write(report)
            f.write("\n" + "-" * 65 + "\n")
            
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ucr_archive_path = os.path.join(base_dir, 'misc', 'data', 'UCRArchive_2018')
    global_results_dir = os.path.join(base_dir, 'results')
    

    if not os.path.exists(ucr_archive_path):
        print(f"ERRO: Pasta {ucr_archive_path} não encontrada.")
        return
        
    datasets = [d for d in os.listdir(ucr_archive_path) if os.path.isdir(os.path.join(ucr_archive_path, d))]
    
    # For testing purposes, we can filter to a specific dataset. Remove this line to process all datasets.
    if DATASET:
        if isinstance(DATASET, list):
            datasets = [d for d in datasets if d in DATASET] * REDUNDANCE
        else:
            datasets = [d for d in datasets if d == DATASET] * REDUNDANCE
    
    print(f"[*] Datasets encontrados: {datasets}")
                        
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] A usar o dispositivo: {device}\n")

    pbar_datasets = tqdm(datasets, desc="Progresso Global (Datasets)", position=0)
    for i, dataset_name in enumerate(pbar_datasets):
        
        redundance_idx = (i // len(DATASET)) + 1 # Calculate the redundancy index
        
        train_path = os.path.join(ucr_archive_path, dataset_name, f'{dataset_name}_TRAIN.tsv')
        test_path = os.path.join(ucr_archive_path, dataset_name, f'{dataset_name}_TEST.tsv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            continue
            
        ds_results_dir = os.path.join(global_results_dir, f'{dataset_name}', f'run_{redundance_idx}')
        os.makedirs(ds_results_dir, exist_ok=True)
        
        X_train, y_train, X_test, y_test, num_classes, orig_classes = load_and_encode_ucr(train_path, test_path)
        seq_length = X_train.shape[1]
        
        # Check if the size of the train set is less than the batch size, and if so, adjust the batch size to avoid errors in DataLoader
        current_batch_size = min(BATCH_SIZE, len(X_train))
        print(f"[*] Verifying batch size compatibility: Train samples = {len(X_train)}, Batch size = {current_batch_size}")
        
        
        train_loader = DataLoader(UCRDatasetWrapper(X_train, y_train, IMG_STRATEGY), batch_size=current_batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(UCRDatasetWrapper(X_test, y_test, IMG_STRATEGY), batch_size=current_batch_size   , shuffle=False)
        
        
      

        # Get the first image directly from the dataset
        # Get the first item from the dataset
        first_item = train_loader.dataset[0]
        
        # Unpack the tuple returned by UCRDatasetWrapper (image, label)
        eg_img = first_item[0]

        # 1. Safely and efficiently convert ONLY the image to a PyTorch tensor
        if not isinstance(eg_img, torch.Tensor):
            eg_img = torch.from_numpy(np.asarray(eg_img).copy())
            
        # 2. Ensure it has exactly 3 dimensions (C, H, W) for the logging statement
        if eg_img.dim() == 2:
            eg_img = eg_img.unsqueeze(0)
            
        # --- INFORMATIONAL LOGGING ---
        print(f"\n\n\n=== Dataset: {dataset_name} ===")
        print(f"· Original Classes: {orig_classes}")
        print(f"· Encoded Classes: {num_classes}")
        print(f"· Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"· Sequence length: {seq_length}")
        print(f"· Image Strategy: {IMG_STRATEGY.upper()}")
        print(f"· Input image size: {eg_img.shape[1]}x{eg_img.shape[2]} (Channels: {eg_img.shape[0]})")
            
        
        # --- TRAINING (calls trainer.py) ---
        results_dict = run_experiment(
            train_loader, test_loader, num_classes, device, ds_results_dir, 
            epochs=EPOCHS,
            models_to_run=MODELS_TO_RUN,
            descriptor_size=DESCRIPTOR_SIZE,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate_xgb=LEARNING_RATE_XGB
        )
        
            
        # --- FECHING RESULTS AND SAVING METRICS ---
        # 1. Save the confusion matrix for each model
        for model_name, result_data in results_dict.items():
            y_true = result_data['y_true']
            y_pred = result_data['y_pred']
            save_confusion_matrix(y_true, y_pred, orig_classes, dataset_name, model_name, ds_results_dir)
        
        # 1.1. Save a .csv with the accuracy metrics for all models and all datasets (for easier plotting later)
        with open(os.path.join(global_results_dir, 'summary_metrics.csv'), 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Timestamp', 'Dataset'] + MODELS_TO_RUN
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:  # Write header only if file is new/empty
                writer.writeheader()
            row = {'Dataset': dataset_name}
            for model_name in MODELS_TO_RUN:
                acc = np.mean(np.array(results_dict[model_name]['y_true']) == np.array(results_dict[model_name]['y_pred']))
                row[model_name] = f"{acc * 100:.2f}%"
                row['Timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                
            writer.writerow(row)
            
        # 2. Computacional Complexity (MACs and Params)
        save_hardware_complexity(results_dict, eg_img.shape[1], ds_results_dir, device)

        # 3. Save a .txt with the accuracy metrics (can be expanded to include more metrics)
        with open(os.path.join(ds_results_dir, 'metrics_summary.txt'), 'w') as f:
            f.write(f"=== Metrics Summary - {dataset_name} ===\n")
            for model_name, result_data in results_dict.items():
                y_true = result_data['y_true']
                y_pred = result_data['y_pred']
                acc = np.mean(np.array(y_true) == np.array(y_pred))
                f.write(f"{model_name}: Acurácia = {acc * 100:.2f}%\n")
        
        # 4. Get the first generated image for each model and save it (for visualization purposes)
        # 1. Fetch the first batch from the dataloader
        first_batch = next(iter(train_loader))

        # Depending on your dataset, the loader might return a tuple of (images, labels) or just images
        if isinstance(first_batch, (list, tuple)):
            images = first_batch[0]
        else:
            images = first_batch

        # 2. Extract the very first image from the batch
        img = images[0]

        # 3. Convert PyTorch tensor to NumPy array
        if hasattr(img, 'detach'):
            img = img.detach().cpu().numpy()

        # 4. Handle dimension ordering (PyTorch C,H,W -> Matplotlib H,W,C)
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:
                img = img.squeeze(axis=-1)

        # 5. Plot the image
        plt.figure(figsize=(4, 4))
        if img.ndim == 2:
            plt.imshow(img, cmap='viridis')
        else:
            plt.imshow(img)

        plt.title('Exemplo de Imagem do Dataset')
        plt.axis('off')
        plt.grid(False) # Turn off the grid specifically for the image

        # 6. Save the figure
        plt.savefig(os.path.join(ds_results_dir, 'dataset_example_image.png'), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()