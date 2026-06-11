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
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Import for hardware complexity analysis (THOP)
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from temporal2img import Temporal2ImageDataset
from trainer import run_experiment



# ---- Global Constants and Configurations ----
# (You can adjust these as needed, or move them to a config file)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MODELS_TO_RUN = ['MobileNet_XGB', 'MobileNetV2', 'EfficientNet']  # Options: 'MobileNetV2', 'EfficientNet', 'MobileNet_XGB'
IMG_STRATEGY = 'gaf'  # Options: 'gaf', 'stft'

# XGBoost Hyperparameters (can be tuned for better performance)
N_ESTIMATORS = 100
MAX_DEPTH = 10
LEARNING_RATE_XGB = 0.1
DESCRIPTOR_SIZE = 1024

DATASET = None  # Change this to the dataset you want to run (e.g., 'InlineSkate', 'ElectricDevices', etc.)


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
    """Calculate and save MACs and Params if THOP is available."""
    if not THOP_AVAILABLE:
        return
        
    filepath = os.path.join(save_dir, 'hardware_complexity_MACs.txt')
    dummy_input = torch.randn(1, 1, seq_length, seq_length).to(device)
    
    with open(filepath, 'w') as f:
        f.write("=== FPGA Hardware Metrics (Thop) ===\n")
        f.write(f"Input size: 1x1x{seq_length}x{seq_length}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<20} | {'MACs (Mega)':<15} | {'Params (Mega)':<15}\n")
        f.write("-" * 60 + "\n")
        
        for model_name, data in models_dict.items():
            model = data['model']
            if model_name == 'MobileNet_XGB':
                macs, params = profile(model.cnn_extractor, inputs=(dummy_input, ), verbose=False)
            else:
                macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
                
            f.write(f"{model_name:<20} | {macs / 1e6:>10.2f} M   | {params / 1e6:>10.2f} M\n")

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
        datasets = [d for d in datasets if d == DATASET] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] A usar o dispositivo: {device}\n")
    
    pbar_datasets = tqdm(datasets, desc="Progresso Global (Datasets)", position=0)
    for dataset_name in pbar_datasets:
        
        train_path = os.path.join(ucr_archive_path, dataset_name, f'{dataset_name}_TRAIN.tsv')
        test_path = os.path.join(ucr_archive_path, dataset_name, f'{dataset_name}_TEST.tsv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            continue
            
        ds_results_dir = os.path.join(global_results_dir, dataset_name)
        os.makedirs(ds_results_dir, exist_ok=True)
        
        X_train, y_train, X_test, y_test, num_classes, orig_classes = load_and_encode_ucr(train_path, test_path)
        seq_length = X_train.shape[1]
        
        train_loader = DataLoader(UCRDatasetWrapper(X_train, y_train, IMG_STRATEGY), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(UCRDatasetWrapper(X_test, y_test, IMG_STRATEGY), batch_size=BATCH_SIZE   , shuffle=False)
        
        
        # --- INFORMATIONAL LOGGING ---
        print(f"\n=== Dataset: {dataset_name} ===")
        print(f"· Original Classes: {orig_classes}")
        print(f"· Encoded Classes: {num_classes}")
        print(f"· Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"· Sequence length: {seq_length}")
        print(f"· Image Strategy: {IMG_STRATEGY.upper()}")
        print(f"· Input image size: {seq_length}x{seq_length}")
        
        
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
        
            
        # 2. Computacional Complexity (MACs and Params)
        save_hardware_complexity(results_dict, seq_length, ds_results_dir, device)

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