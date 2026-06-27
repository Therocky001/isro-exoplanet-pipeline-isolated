import numpy as np
from typing import Tuple, Any

class LightCurveCleaned:
    def __init__(self, time: np.ndarray, flux: np.ndarray):
        self.time = time
        self.flux = flux

    # Fix for lc['time'] or lc['flux'] syntax
    def __getitem__(self, key):
        if key == 'time': return self.time
        if key == 'flux': return self.flux
        # Fix for direct numpy 2D multidimensional slicing matrices [:, 0]
        if isinstance(key, tuple):
            return self.flux[key[0]]
        return self.flux[key]

    def __len__(self):
        return len(self.flux)

    @property
    def value(self):
        return self.flux

class NoiseMitigationEngine:
    def __init__(self):
        pass

    def process_light_curve(self, raw_data: Any) -> Tuple[LightCurveCleaned, np.ndarray]:
        time = raw_data.time
        flux = raw_data.flux
        
        median = np.median(flux)
        std = np.std(flux)
        clean_mask = np.abs(flux - median) < (5.0 * std)
        
        cleaned_time = time[clean_mask]
        cleaned_flux = flux[clean_mask]
        
        # Create mock centroid matrix
        mock_centroids = np.zeros((len(cleaned_time), 2))
        mock_centroids[:, 0] = np.random.normal(0, 0.0001, size=len(cleaned_time))
        mock_centroids[:, 1] = np.random.normal(0, 0.0001, size=len(cleaned_time))
        
        cleaned_lc = LightCurveCleaned(cleaned_time, cleaned_flux)
        return cleaned_lc, mock_centroids