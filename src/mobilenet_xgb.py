# /*****************************************************************************/
#  * File: mobilenet_xgb.py
#  * Author: Olavo Alves Barros Silva
#  * Description: Hybrid Pipeline (CNN + XGBoost) optimized for FPGA.
#  * Uses a modified MobileNetV2 to extract a highly compressed descriptor
#  * (e.g., 64 features) to feed an Extreme Gradient Boosting classifier.
# /*****************************************************************************/

import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import os

# Imports the base MobileNetV2 created previously
try:
    from mobilenetv2 import MobileNetV2
except ImportError:
    raise ImportError("Make sure mobilenet_v2.py is in the same directory.")

class CompressedFeatureExtractor(nn.Module):
    """
    CNN branch that compresses feature maps into a small descriptor.
    Ideal for saving BRAMs on the FPGA when passing data to XGBoost.
    """
    def __init__(self, num_classes, in_channels=1, descriptor_size=64, width_mult=0.5):
        super(CompressedFeatureExtractor, self).__init__()
        
        # 1. Load the backbone with a width multiplier of 0.5 to save MACs on the FPGA
        self.backbone = MobileNetV2(num_classes=num_classes, in_channels=in_channels, width_mult=width_mult)
        
        # We replace the original classifier with a projection (reduction) layer
        # If width_mult=0.5, self.backbone.last_channel will be 640.
        backbone_out_features = self.backbone.last_channel
        
        # 2. Projection Layer: Compresses from 640 to the descriptor size (e.g., 64)
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_features, descriptor_size),
            nn.BatchNorm1d(descriptor_size),
            nn.ReLU6(inplace=True) 
        )
        
        # 3. Temporary Classifier (Used ONLY to train the CNN)
        self.temp_classifier = nn.Linear(descriptor_size, num_classes)

    def forward(self, x, extract_features=False):
        # Pass through the CNN (returns the vector already with Global Average Pooling)
        features = self.backbone(x, extract_features=True) 
        
        # Compress the features (E.g.: [Batch, 640] -> [Batch, 64])
        compressed_descriptor = self.projection(features)
        
        if extract_features:
            return compressed_descriptor
            
        # Output for CrossEntropyLoss (CNN training phase)
        return self.temp_classifier(compressed_descriptor)


class PipelineMobileNetXGB:
    def __init__(self, num_classes=3, in_channels=1, descriptor_size=64, device='cpu',
                 n_estimators=100, max_depth=5, learning_rate=0.1):
        """
        Main class that manages the training of the CNN and XGBoost.
        """
        self.device = device
        self.descriptor_size = descriptor_size
        
        # Instantiate the CNN with the compression layer
        self.cnn_extractor = CompressedFeatureExtractor(
            num_classes=num_classes, 
            in_channels=in_channels, 
            descriptor_size=descriptor_size,
            width_mult=0.5 # Cuts the CNN in half for the FPGA
        ).to(self.device)
        
        # Instantiate XGBoost (Hyperparameters controlled to prevent giant trees)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,        # Number of trees
            max_depth=max_depth,             # Max depth (4 is FPGA-friendly)
            learning_rate=learning_rate,
            objective='multi:softmax',
            num_class=num_classes,
            eval_metric='mlogloss',
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
        """
        Extracts features from the trained CNN and trains XGBoost.
        """
        # 1. Feature Extraction
        print("\n--- Extraction Phase for XGBoost ---")
        X_train, y_train = self.extract_dataset_features(train_loader)
        
        # 2. XGBoost Training
        print("Training XGBoost Decision Trees...")
        self.xgb_model.fit(X_train, y_train)
        print("[OK] XGBoost Trained.")
        
        # 3. Quick evaluation if test data is provided
        # if test_loader is not None:
        #     X_test, y_test = self.extract_dataset_features(test_loader)
        #     y_pred = self.xgb_model.predict(X_test)
        #     acc = accuracy_score(y_test, y_pred)
        #     print(f"\nXGBoost Test Accuracy: {acc * 100:.2f}%")
        #     print("\nClassification Report:\n", classification_report(y_test, y_pred))

    def save_pipeline(self, save_dir):
        """Saves the CNN weights and the XGBoost model"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the CNN
        torch.save(self.cnn_extractor.state_dict(), os.path.join(save_dir, 'cnn_extractor.pth'))
        
        # Save XGBoost (.json format is very useful for parsing later into C++/FPGA)
        self.xgb_model.save_model(os.path.join(save_dir, 'xgboost_model.json'))
        print(f"Pipeline saved in: {save_dir}")

