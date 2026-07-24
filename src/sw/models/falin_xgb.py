# /*****************************************************************************/
#  * File: falin_xgb.py
#  * Author: Olavo Alves Barros Silva
#  * Description: Hybrid Pipeline: FALIN CNN (Feature Extractor) + 
#  * XGBoost (Classifier) optimized for FPGA deployment.
# /*****************************************************************************/

from models.falin_cnn import FalinCNN
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import os

class PipelineFalinXGB:
    """
    Hybrid Pipeline: FALIN CNN (Feature Extractor) + XGBoost (Classifier)
    """
    def __init__(self, num_classes, in_channels=1, descriptor_size=64, device='cpu', n_estimators=100, max_depth=3, learning_rate=0.1):
        self.device = device
        self.num_classes = num_classes
        
        # Instantiate the lightweight CNN
        self.cnn_extractor = FalinCNN(num_classes=num_classes, 
                                      in_channels=in_channels, 
                                      descriptor_size=descriptor_size).to(self.device)
        
        # Optimized XGBoost (shallow trees are better for hardware translation)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth, # Keep it low (3 to 5) to avoid generating massive hardware logic
            learning_rate=learning_rate,
            objective='multi:softprob' if self.num_classes > 2 else 'binary:logistic',
            eval_metric='mlogloss',
            # use_label_encoder=False,
            random_state=42
        )

    def extract_dataset_features(self, dataloader):
        """
        Passes the entire dataset through the CNN (without gradients) to generate 
        the matrix (N_samples, 64_features) for XGBoost.
        """
        self.cnn_extractor.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                
                # Extract_features=True bypasses the CNN's Linear layer
                features = self.cnn_extractor(inputs, extract_features=True)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                
        X_numpy = np.concatenate(all_features)
        y_numpy = np.concatenate(all_labels)
        
        return X_numpy, y_numpy

    def train_xgboost(self, train_loader, test_loader=None):
        print("\n--- Extraction Phase for XGBoost ---")
        X_train, y_train = self.extract_dataset_features(train_loader)
        
        print("Training XGBoost Decision Trees...")
        self.xgb_model.fit(X_train, y_train)
        print("[OK] XGBoost Trained.")
        
        # if test_loader is not None:
        #     X_test, y_test = self.extract_dataset_features(test_loader)
        #     y_pred = self.xgb_model.predict(X_test)
        #     acc = accuracy_score(y_test, y_pred)
        #     print(f"\nXGBoost Test Accuracy: {acc * 100:.2f}%")
        #     print("\nClassification Report:\n", classification_report(y_test, y_pred))