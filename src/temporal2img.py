# /*****************************************************************************/
#  * File: temporal2img.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-05-16
#  * License: MIT License
#  * Description: This class is a data loader responsable to transform temporal 
#  * data into images, by using different techniques,  such as 
#  * Gramian Angular Fields (GAF) and Short-Time Fourier Transform (STFT).
#  * # /*****************************************************************************/

import numpy as np
import torch
import scipy.signal
from torch.utils.data import DataLoader, Dataset


class Temporal2ImageDataset(Dataset):
    def __init__(self, temporal_data, transform=None, estrategy='gaf'):
        self.temporal_data = temporal_data
        self.transform = transform
        self.estrategy = estrategy.lower()

    def __len__(self):
        return len(self.temporal_data)

    def __getitem__(self, idx):
        temporal_sample = self.temporal_data[idx]
        
        # Transform the temporal data into an image
        image_sample = self.temporal_to_image(temporal_sample)
        
        # Apply standard torchvision transforms if provided (e.g., ToTensor)
        if self.transform:
            image_sample = self.transform(image_sample)
        
        return image_sample

    def temporal_to_image(self, temporal_sample):
        """Transforms 1D temporal data into a 2D image representation based on the chosen strategy.

        Args:
            temporal_sample (numpy.ndarray or torch.Tensor): 1D temporal data array.
            
        Returns:
            numpy.ndarray: 2D image representation (e.g., GAF or STFT).
        """
        # Ensure it's a numpy array for mathematical operations
        if isinstance(temporal_sample, torch.Tensor):
            temporal_sample = temporal_sample.numpy()
            
        temporal_sample = np.asarray(temporal_sample).flatten()

        if self.estrategy == 'gaf':
            return self._gramian_encoding(temporal_sample)
        elif self.estrategy == 'stft':
            return self._stft_encoding(temporal_sample)
        else:
            raise ValueError(f"Strategy '{self.estrategy}' is not supported. Try 'gaf' or 'stft'.")

    def _piecewise_aggregate_approximation(self, temporal_sample):
        """Piecewise Aggregate Approximation (PAA)
            
        
        """

    def _gramian_encoding(self, temporal_sample):
        """This function encodes the temporal data into an image using Gramian Angular Summation Fields (GASF).

        It is calculated as:
        $GAF(i, j) = cos(\phi_i + \phi_j)$
        where $\phi_i$ is the angle corresponding to the temporal data point $i$.

        Args:
            temporal_sample (numpy.ndarray): 1D array of temporal data.
            
        Returns:
            numpy.ndarray: 2D Gramian Angular Field (shape: N x N, where N is the sequence length).
        """
        # 1. Normalize the time series to the range [-1, 1]
        min_val = np.min(temporal_sample)
        max_val = np.max(temporal_sample)
        
        if max_val == min_val:
            # Handle edge case where the signal is perfectly flat (constant)
            normalized_sample = np.zeros_like(temporal_sample)
        else:
            normalized_sample = ((temporal_sample - min_val) / (max_val - min_val)) * 2 - 1
            
        # 2. Transform to polar coordinates (calculate the angle phi)
        # Clip strictly between [-1, 1] to avoid numpy numerical instability during arccos
        normalized_sample = np.clip(normalized_sample, -1.0, 1.0)
        phi = np.arccos(normalized_sample)
        
        # 3. Calculate the Gramian Angular Summation Field (GASF)
        # By trigonometric identity: cos(A + B) = cos(A)cos(B) - sin(A)sin(B)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Use outer product to rapidly generate the N x N image matrix without loop overhead
        gaf_matrix = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
        
        # Return as float32 for PyTorch compatibility
        return gaf_matrix.astype(np.float32)

    def _stft_encoding(self, temporal_sample, fs=1.0, nperseg=64):
        """Encodes temporal data into a 2D spectrogram using Short-Time Fourier Transform (STFT).

        Args:
            temporal_sample (numpy.ndarray): 1D array of temporal data.
            fs (float): Sampling frequency.
            nperseg (int): Length of each segment (window size).
            
        Returns:
            numpy.ndarray: 2D Spectrogram (Magnitude of STFT).
        """
        # Ensure the window segment is not larger than the sample itself
        nperseg = min(nperseg, len(temporal_sample))
        
        # Apply scipy's STFT
        f, t, Zxx = scipy.signal.stft(temporal_sample, fs=fs, nperseg=nperseg)
        
        # Get the magnitude (absolute value) of the complex return to create the image
        spectrogram = np.abs(Zxx)
        
        return spectrogram.astype(np.float32)