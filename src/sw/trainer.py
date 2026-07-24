# /*****************************************************************************/
#  * File: trainer.py
#  * Author: Olavo Alves Barros Silva
#  * Description: Module responsible for training and evaluating all models 
#  * (FalinCNN, MobileNetV2, EfficientNet, and their Hybrid XGBoost pipelines)
#  * for a specific dataset.
# /*****************************************************************************/

import os
import time
import copy
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# Import the new lightweight custom models (Hardware Optimized)
from models.falin_cnn import FalinCNN
from models.falin_xgb import PipelineFalinXGB

# Import the old models (Baselines for comparison)
try:
    from models.mobilenetv2 import MobileNetV2
    from models.efficient import EfficientNetCustom
    from models.mobilenet_xgb import PipelineMobileNetXGB
except ImportError:
    print("[Warning] Baseline models (MobileNet/EfficientNet) not found. Make sure their files exist if you want to run them.")


import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_standard_model(model, train_loader, test_loader, device, model_name, scheduler_patience, early_stop_patience, epochs=5, tb_writer=None, lr=1e-4):
    """
    Trains standard end-to-end CNN models with Early Stopping, Best Model Checkpointing,
    and a Learning Rate Scheduler (ReduceLROnPlateau).
    """
    if model is None:
        raise ValueError("Model is None. Cannot create optimizer.")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    try:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    except Exception as e:
        raise RuntimeError(f"Failed to create optimizer: {e}")
        
    # --- INITIALIZE SCHEDULER ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=scheduler_patience, min_lr=1e-6
    )
    
    best_acc = 0.0
    # Create a copy of the initial weights to avoid returning a referenced corrupted state
    best_model_wts = copy.deepcopy(model.state_dict()) 
    epochs_no_improve = 0 # Early stopping counter
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    pbar_epochs = tqdm(range(epochs), desc=f"[{model_name}] Training", leave=True)
    
    for epoch in pbar_epochs:
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_running_loss / len(test_loader.dataset)
        epoch_val_acc = val_correct / val_total
        history['test_loss'].append(epoch_val_loss)
        history['test_acc'].append(epoch_val_acc)

        # --- STEP THE SCHEDULER ---
        # ReduceLROnPlateau requires the metric it is monitoring to be passed in.
        scheduler.step(epoch_val_loss)
        
        # --- BEST MODEL CHECKPOINT & EARLY STOPPING LOGIC ---
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict()) # Deep copy the model weights
            epochs_no_improve = 0 # Reset patience counter
        else:
            epochs_no_improve += 1
            
        # --- TENSORBOARD LOGGING ---
        if tb_writer:
            tb_writer.add_scalars('Loss', {'Train': epoch_train_loss, 'Validation': epoch_val_loss}, epoch)
            tb_writer.add_scalars('Accuracy', {'Train': epoch_train_acc, 'Validation': epoch_val_acc}, epoch)
            
            # Fetch the updated learning rate (useful to track scheduler adjustments)
            current_lr = optimizer.param_groups[0]['lr']
            tb_writer.add_scalar('Hyperparameters/Learning_Rate', current_lr, epoch)
            
            for name, param in model.named_parameters():
                tb_writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                if param.grad is not None:
                    tb_writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            
        pbar_epochs.set_postfix({
            'T_Loss': f"{epoch_train_loss:.4f}", 
            'V_Loss': f"{epoch_val_loss:.4f}", 
            'V_Acc': f"{epoch_val_acc*100:.2f}%",
            'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
        # Break the loop if patience is reached
        if epochs_no_improve >= early_stop_patience:
            print(f"\n[Early Stopping] Triggered at epoch {epoch}. No improvement for {early_stop_patience} epochs.")
            break
            
    print(f"\n[{model_name}] Training completed. Best Validation Accuracy: {best_acc*100:.2f}%")
    
    # --- LOAD BEST WEIGHTS BEFORE RETURNING ---
    model.load_state_dict(best_model_wts)
    
    
    return model, best_acc, history

def run_experiment(train_loader, test_loader, num_classes, device, results_dir,
                   dataset_name, epochs, models_to_run, descriptor_size, learning_rate,
                   n_estimators, max_depth, learning_rate_xgb,
                   scheduler_patience, early_stop_patience, in_channels=1,
                   img_strategy='GAF', seq_length=0, orig_classes=[0],
                   ):
    """
    Main function to orchestrate the training of selected models.
    """
    results = {}
    # 2026-07-23 - Olavo: dataset_name is now passed explicitly instead of being
    # inferred from os.path.basename(results_dir), which returned "run_N" (the
    # results_dir's last path component) instead of the actual dataset name,
    # since results_dir is .../<dataset_name>/run_N.


    # Função auxiliar para extrair previsões de modelos PyTorch padrão
    def get_predictions(model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        return all_labels, all_preds
    
    # 2026-07-23 10:21:46 - Olavo: New aux function to save checkpoint
    def save_model_checkpoint(model_to_save, model_name):
        checkpoint_path = os.path.join(results_dir, f"{model_name}_checkpoint.pth")
        torch.save({
            'state_dict': model_to_save.state_dict(),
            'dataset': dataset_name,
            'img_strategy': img_strategy,
            'descriptor_size': descriptor_size,
            'seq_length': seq_length,
            'num_classes': num_classes,
            'label_classes': list(orig_classes),
        }, checkpoint_path)
        print(f"[{model_name}] Checkpoint salvo em: {checkpoint_path}")

    # =========================================================================
    # NEW MODELS (TINYML / FPGA OPTIMIZED)
    # =========================================================================

    if 'Falin_XGB' in models_to_run:
        print("\n==================================================")
        print(f"Starting FALIN CNN + XGBoost pipeline for {dataset_name}")
        print("==================================================")
        
        tb_writer_xgb = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard", "Falin_XGB"))
        
        pipeline_xgb = PipelineFalinXGB(
            num_classes=num_classes, in_channels=in_channels, 
            descriptor_size=descriptor_size, device=device, 
            n_estimators=n_estimators, max_depth=max_depth, 
            learning_rate=learning_rate_xgb
        )
        
        print("[Falin_XGB] Pre-training the CNN extractor...")
        pipeline_xgb.cnn_extractor, _, _ = train_standard_model(
            model=pipeline_xgb.cnn_extractor,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            model_name="CNN_Extractor (Falin)",
            scheduler_patience=scheduler_patience,
            early_stop_patience=early_stop_patience,
            epochs=epochs,
            tb_writer=tb_writer_xgb,
            lr=learning_rate
        )
        tb_writer_xgb.close()
        
        # 2026-07-23 10:23:08 - Olavo : Saving the model
        save_model_checkpoint(pipeline_xgb.cnn_extractor, "Falin_XGB_CNN_Extractor")
        
        pipeline_xgb.train_xgboost(train_loader, test_loader)
        
        # Extrair previsões para devolver à main.py
        X_test, y_test = pipeline_xgb.extract_dataset_features(test_loader)
        y_pred = pipeline_xgb.xgb_model.predict(X_test)
        
        results['Falin_XGB'] = {'y_true': y_test, 'y_pred': y_pred, 'model': pipeline_xgb}

    if 'FalinCNN' in models_to_run:
        print("\n==================================================")
        print(f"Starting standalone FALIN CNN for {dataset_name}")
        print("==================================================")
        
        tb_writer_cnn = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard", "FalinCNN"))
        model_cnn = FalinCNN(num_classes=num_classes, in_channels=in_channels, descriptor_size=descriptor_size).to(device)
        model_cnn, best_acc, _ = train_standard_model(
            model=model_cnn,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            model_name="FalinCNN",
            scheduler_patience=scheduler_patience,
            early_stop_patience=early_stop_patience,
            epochs=epochs,
            tb_writer=tb_writer_cnn,
            lr=learning_rate
        )
        tb_writer_cnn.close()
        
        save_model_checkpoint(model_cnn, "FalinCNN")
        
        y_true, y_pred = get_predictions(model_cnn, test_loader)
        results['FalinCNN'] = {'y_true': y_true, 'y_pred': y_pred, 'model': model_cnn}

    # 2026-07-22 21:44:00 - Olavo : Add Faling 

    # =========================================================================
    # LEGACY BASELINE MODELS
    # =========================================================================

    if 'MobileNet_XGB' in models_to_run:
        print("\n==================================================")
        print(f"Starting MobileNetV2 + XGBoost pipeline for {dataset_name}")
        print("==================================================")
        
        tb_writer_mxgb = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard", "MobileNet_XGB"))
        
        pipeline_mxgb = PipelineMobileNetXGB(
            num_classes=num_classes, in_channels=in_channels, 
            descriptor_size=descriptor_size, device=device, 
            n_estimators=n_estimators, max_depth=max_depth, 
            learning_rate=learning_rate_xgb
        )
        
        print("[MobileNet_XGB] Pre-training the CNN extractor...")
        pipeline_mxgb.cnn_extractor, _, _ = train_standard_model(
            model=pipeline_mxgb.cnn_extractor,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            model_name="CNN_Extractor (MobileNet)",
            scheduler_patience=scheduler_patience,
            early_stop_patience=early_stop_patience,
            epochs=epochs,
            tb_writer=tb_writer_mxgb,
            lr=learning_rate
        )
        tb_writer_mxgb.close()
        
        
        pipeline_mxgb.train_xgboost(train_loader, test_loader)
        
        X_test, y_test = pipeline_mxgb.extract_dataset_features(test_loader)
        y_pred = pipeline_mxgb.xgb_model.predict(X_test)
        
        results['MobileNet_XGB'] = {'y_true': y_test, 'y_pred': y_pred, 'model': pipeline_mxgb}

    if 'MobileNetV2' in models_to_run:
        print("\n==================================================")
        print(f"Starting standalone MobileNetV2 for {dataset_name}")
        print("==================================================")
        
        tb_writer_mb = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard", "MobileNetV2"))
        model_mb = MobileNetV2(num_classes=num_classes, in_channels=in_channels).to(device)
        model_mb, best_acc, _ = train_standard_model(
            model=model_mb,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            model_name="MobileNetV2",
            scheduler_patience=scheduler_patience,
            early_stop_patience=early_stop_patience,
            epochs=epochs,
            tb_writer=tb_writer_mb,
            lr=learning_rate
        )
        tb_writer_mb.close()
        
        y_true, y_pred = get_predictions(model_mb, test_loader)
        results['MobileNetV2'] = {'y_true': y_true, 'y_pred': y_pred, 'model': model_mb}

    return results