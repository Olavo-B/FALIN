# /*****************************************************************************/
#  * File: trainer.py
#  * Author: Olavo Alves Barros Silva
#  * Description: Módulo responsável por treinar e avaliar os modelos 
#  * (MobileNetV2, EfficientNet e MobileNet+XGBoost) para um dataset específico.
# /*****************************************************************************/

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time

# Importação dos modelos que você já criou
from mobilenetv2 import MobileNetV2
from efficient import EfficientNetCustom
from mobilenet_xgb import PipelineMobileNetXGB

def train_standard_model(model, train_loader, test_loader, device, model_name, epochs=5, tb_writer=None):
    """
    Treina modelos CNN padrão (MobileNetV2 e EfficientNet).
    """
    criterion = nn.CrossEntropyLoss()
    # Ensure model is on the correct device before creating the optimizer
    if model is None:
        raise ValueError("Model is None. Cannot create optimizer.")
    model.to(device)
    try:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    except Exception as e:
        # Provide a clearer error instead of a low-level segfault
        raise RuntimeError(f"Failed to create optimizer: {e}")
    
    
    
    # Progresso por Épocas (A barra enche a cada época completada)
    pbar_epochs = tqdm(range(epochs), desc=f"[{model_name}] A treinar", leave=False)
    
    for epoch in pbar_epochs:
        model.train()
        running_loss = 0.0
        
        # Iteração sobre os batches silenciosamente (sem barra individual)
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.unsqueeze(1).to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Calcula a perda média da época
        avg_loss = running_loss / len(train_loader)
        
        # Atualiza a barra de épocas com a perda atual
        pbar_epochs.set_postfix({'loss': f"{avg_loss:.4f}"})
            
        # Log da Loss no TensorBoard
        if tb_writer:
            tb_writer.add_scalar('Loss/train', avg_loss, epoch)

    # Avaliação no Teste (Ocorre após todas as épocas)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.unsqueeze(1).to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    # Guarda o histograma de pesos no final do treino
    if tb_writer:
        for name, param in model.named_parameters():
            tb_writer.add_histogram(f'Pesos/{name}', param.clone().cpu().data.numpy(), epochs)
    
    return model, all_targets, all_preds

def run_experiment(train_loader, test_loader, num_classes, device, results_dir, 
                   epochs=5, models_to_run=['MobileNetV2', 'EfficientNet', 'MobileNet_XGB'],
                   descriptor_size=256, n_estimators=100, max_depth=5, learning_rate_xgb=0.1):
    """
    Executa o treinamento de TODOS os modelos e retorna as previsões para gerar as métricas.
    """
    results = {}

    if 'MobileNetV2' in models_to_run:
        # ---------------------------------------------------------
        # 1. MobileNetV2
        # ---------------------------------------------------------
        tb_writer_mb = SummaryWriter(log_dir=f"{results_dir}/tensorboard/MobileNetV2")
        mobilenet = MobileNetV2(num_classes=num_classes, in_channels=1)
        mobilenet, y_true_mb, y_pred_mb = train_standard_model(
            mobilenet, train_loader, test_loader, device, "MobileNetV2", epochs, tb_writer_mb
        )
        tb_writer_mb.close()
        results['MobileNetV2'] = {'y_true': y_true_mb, 'y_pred': y_pred_mb, 'model': mobilenet}
        
        
    if 'EfficientNet' in models_to_run:
        # ---------------------------------------------------------
        # 2. EfficientNet
        # ---------------------------------------------------------
        tb_writer_ef = SummaryWriter(log_dir=f"{results_dir}/tensorboard/EfficientNet")
        efficientnet = EfficientNetCustom(num_classes=num_classes, in_channels=1)
        efficientnet, y_true_ef, y_pred_ef = train_standard_model(
            efficientnet, train_loader, test_loader, device, "EfficientNet", epochs, tb_writer_ef
        )
        tb_writer_ef.close()
        results['EfficientNet'] = {'y_true': y_true_ef, 'y_pred': y_pred_ef, 'model': efficientnet}
    if 'MobileNet_XGB' in models_to_run:
        # ---------------------------------------------------------
        # 3. MobileNet + XGBoost
        # ---------------------------------------------------------
        # O pipeline treina a CNN extratora primeiro e depois ajusta o XGBoost
        tb_writer_xgb = SummaryWriter(log_dir=f"{results_dir}/tensorboard/MobileNet_XGB")
        pipeline_xgb = PipelineMobileNetXGB(num_classes=num_classes, in_channels=1, 
                                            descriptor_size=descriptor_size, device=device, 
                                            n_estimators=n_estimators, max_depth=max_depth, 
                                            learning_rate=learning_rate_xgb)
        
        # Treina o extrator da CNN
        pipeline_xgb.cnn_extractor, _, _ = train_standard_model(
            pipeline_xgb.cnn_extractor, train_loader, test_loader, device, "CNN_Extractor (XGB)", epochs, tb_writer_xgb
        )
        tb_writer_xgb.close()
        
        # Extrai features e treina XGBoost
        print("[MobileNet_XGB] A treinar árvores de decisão...")
        X_train_features, y_train_features = pipeline_xgb.extract_dataset_features(train_loader)
        pipeline_xgb.xgb_model.fit(X_train_features, y_train_features)
        
        # Previsões de Teste
        X_test_features, y_test_features = pipeline_xgb.extract_dataset_features(test_loader)
        y_pred_xgb = pipeline_xgb.xgb_model.predict(X_test_features)
        
        results['MobileNet_XGB'] = {'y_true': y_test_features, 'y_pred': y_pred_xgb, 'model': pipeline_xgb}
        
    return results