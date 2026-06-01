# /*****************************************************************************/
#  * File: main.py
#  * Author: Olavo Alves Barros Silva
#  * Description: Orquestrador principal. Itera sobre a base UCR, treina 
#  * modelos, calcula MACs, salva Matrizes de Confusão e logs TensorBoard.
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

# Import cálculo de Hardware (MACs/Params)
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from temporal2img import Temporal2ImageDataset
from trainer import run_experiment

# Wrapper do Dataset para retornar Imagem GAF e Label
class UCRDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, X, y, strategy='gaf'):
        self.X_dataset = Temporal2ImageDataset(X, estrategy=strategy)
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_dataset[idx], self.y[idx]

def load_and_encode_ucr(train_path, test_path):
    """Carrega dados e normaliza os labels de 0 a Num_Classes-1."""
    train_data = np.loadtxt(train_path, delimiter='\t')
    test_data = np.loadtxt(test_path, delimiter='\t')
    
    X_train, y_train_raw = train_data[:, 1:], train_data[:, 0]
    X_test, y_test_raw = test_data[:, 1:], test_data[:, 0]
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    return X_train, y_train, X_test, y_test, len(le.classes_), le.classes_

def save_confusion_matrix(y_true, y_pred, classes, dataset_name, model_name, save_dir):
    """Gera e salva a Matriz de Confusão em formato PNG."""
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
    """Calcula e salva os MACs e Params se THOP estiver disponível."""
    if not THOP_AVAILABLE:
        return
        
    filepath = os.path.join(save_dir, 'hardware_complexity_MACs.txt')
    dummy_input = torch.randn(1, 1, seq_length, seq_length).to(device)
    
    with open(filepath, 'w') as f:
        f.write("=== FPGA Hardware Metrics (Thop) ===\n")
        f.write(f"Tamanho da Imagem de Entrada GAF: 1x1x{seq_length}x{seq_length}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Modelo':<20} | {'MACs (Mega)':<15} | {'Params (Mega)':<15}\n")
        f.write("-" * 60 + "\n")
        
        for model_name, data in models_dict.items():
            model = data['model']
            # Para o pipeline XGB, precisamos passar só o extrator CNN para o thop
            if model_name == 'MobileNet_XGB':
                macs, params = profile(model.cnn_extractor, inputs=(dummy_input, ), verbose=False)
            else:
                macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
                
            f.write(f"{model_name:<20} | {macs / 1e6:>10.2f} M   | {params / 1e6:>10.2f} M\n")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ucr_archive_path = os.path.join(base_dir, 'misc', 'data', 'UCRArchive_2018')
    global_results_dir = os.path.join(base_dir, 'results')
    
    # Lista todos os datasets disponíveis na pasta
    if not os.path.exists(ucr_archive_path):
        print(f"ERRO: Pasta {ucr_archive_path} não encontrada.")
        return
        
    # Pega apenas as pastas (ignorando arquivos como .zip)
    datasets = [d for d in os.listdir(ucr_archive_path) if os.path.isdir(os.path.join(ucr_archive_path, d))]
    
    # Para testes iniciais, você pode querer forçar apenas um dataset. 
    # Comente a linha abaixo para rodar em todos.
    datasets = [d for d in datasets if d == 'CBF'] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] A usar o dispositivo: {device}\n")
    
    # Loop Principal iterando sobre os Datasets
    pbar_datasets = tqdm(datasets, desc="Progresso Global (Datasets)", position=0)
    for dataset_name in pbar_datasets:
        
        train_path = os.path.join(ucr_archive_path, dataset_name, f'{dataset_name}_TRAIN.tsv')
        test_path = os.path.join(ucr_archive_path, dataset_name, f'{dataset_name}_TEST.tsv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            continue
            
        # Cria a subpasta de resultados exclusiva para este dataset
        ds_results_dir = os.path.join(global_results_dir, dataset_name)
        os.makedirs(ds_results_dir, exist_ok=True)
        
        # Carrega e processa os dados
        X_train, y_train, X_test, y_test, num_classes, orig_classes = load_and_encode_ucr(train_path, test_path)
        seq_length = X_train.shape[1]
        
        train_loader = DataLoader(UCRDatasetWrapper(X_train, y_train, 'gaf'), batch_size=16, shuffle=True)
        test_loader = DataLoader(UCRDatasetWrapper(X_test, y_test, 'gaf'), batch_size=16   , shuffle=False)
        
        # --- TREINAMENTO (Chama o trainer.py) ---
        # Definindo poucas épocas por padrão para teste. Suba para 30~50 num treino final.
        results_dict = run_experiment(
            train_loader, test_loader, num_classes, device, ds_results_dir, epochs=10,
            models_to_run=['MobileNetV2']
        )
        
        # --- SALVAR ARTEFATOS ---
        for model_name, result_data in results_dict.items():
            # 1. Salvar Matriz de Confusão
            save_confusion_matrix(
                result_data['y_true'], 
                result_data['y_pred'], 
                orig_classes, 
                dataset_name, 
                model_name, 
                ds_results_dir
            )
            
        # 2. Salvar Complexidade Computacional MACs
        save_hardware_complexity(results_dict, seq_length, ds_results_dir, device)

        # 3. Salvar um .txt com as métricas de acurácia (pode ser expandido para incluir mais métricas)
        with open(os.path.join(ds_results_dir, 'metrics_summary.txt'), 'w') as f:
            f.write(f"=== Resumo de Métricas - {dataset_name} ===\n")
            for model_name, result_data in results_dict.items():
                y_true = result_data['y_true']
                y_pred = result_data['y_pred']
                acc = np.mean(np.array(y_true) == np.array(y_pred))
                f.write(f"{model_name}: Acurácia = {acc * 100:.2f}%\n")
        

if __name__ == '__main__':
    main()