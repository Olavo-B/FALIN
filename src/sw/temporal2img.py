# /*****************************************************************************/
#  * File: temporal2img.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-06-24
#  * Description: This class is a data loader responsible for transforming 
#  * temporal data into images using techniques such as GAF and STFT.
#  * It pre-computes and caches images in RAM to maximize GPU utilization.
# /*****************************************************************************/

import numpy as np
import torch
import scipy.signal
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

class Temporal2ImageDataset(Dataset):
    def __init__(self, temporal_data, target_size=(96, 96), estrategy='gaf'):
        """
        Args:
            temporal_data (array-like): The 1D time-series dataset.
            target_size (tuple): Fixed (Height, Width) for the output images.
            estrategy (str): Encoding strategy ('gaf' or 'stft').
        """
        self.estrategy = estrategy.lower()
        self.target_size = target_size
        
        print(f"\n[*] Pre-computing {self.estrategy.upper()} images into RAM. Please wait...")
        
        precomputed_images = []
        
        # Pre-compute all images and resize them once to avoid CPU bottlenecks during training
        for sample in tqdm(temporal_data, desc="Transforming 1D to 2D"):
            # 1. Generate the image matrix at full resolution
            img_matrix = self.temporal_to_image(sample)
            img_tensor = torch.tensor(img_matrix, dtype=torch.float32)
            
            # 2. Add dimensions for interpolation: (H, W) -> (1, 1, H, W)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            
            # 3. Resize using bilinear interpolation
            img_resized = F.interpolate(
                img_tensor, 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # 4. Squeeze back to (1, H_target, W_target)
            img_resized = img_resized.squeeze(0)
            
            precomputed_images.append(img_resized)
            
        # Stack into a single contiguous tensor in RAM
        self.images = torch.stack(precomputed_images)
        print(f"[*] Dataset loaded in RAM! Shape: {self.images.shape}\n")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # DataLoader returns the precomputed image instantly from RAM
        return self.images[idx]

    def temporal_to_image(self, temporal_sample):
        """ Transforms a 1D sample into a 2D matrix. """
        if isinstance(temporal_sample, torch.Tensor):
            temporal_sample = temporal_sample.numpy()
            
        temporal_sample = np.asarray(temporal_sample).flatten()
        
        if self.estrategy == 'gaf':
            return self._gramian_encoding(temporal_sample)
        elif self.estrategy == 'stft':
            return self._stft_encoding(temporal_sample)
        else:
            raise ValueError(f"Strategy '{self.estrategy}' not supported.")

    def _gramian_encoding(self, temporal_sample):
        max_val = np.max(temporal_sample)
        min_val = np.min(temporal_sample)
        
        if max_val == min_val:
            normalized_sample = np.zeros_like(temporal_sample)
        else:
            normalized_sample = ((temporal_sample - min_val) / (max_val - min_val)) * 2 - 1
            
        x_norm = np.clip(normalized_sample, -1.0, 1.0)
        
        cos_phi = x_norm
        sin_phi = np.sqrt(1.0 - x_norm ** 2)
        
        gaf_matrix = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
        return gaf_matrix.astype(np.float32)

    def _stft_encoding(self, temporal_sample, fs=1.0, nperseg=64):
        nperseg = min(nperseg, len(temporal_sample))
        f, t, Zxx = scipy.signal.stft(temporal_sample, fs=fs, nperseg=nperseg, return_onesided=True)
        magnitude = np.abs(Zxx)
        return magnitude.astype(np.float32)