import numpy as np
from typing import List, Optional

class LightCurveData:
    def __init__(self, time: np.ndarray, flux: np.ndarray):
        self.time = time
        self.flux = flux

class DataIngestion:
    def fetch_light_curve(self, target_id: str, sectors: Optional[List[int]] = None, cadence: str = "short") -> LightCurveData:
        # Hackathon mock/simulation generator for testing connection orchestration
        # Real implementation uses lightkurve.search_lightcurve
        np.random.seed(42)
        time = np.linspace(0, 27.4, 3000)  # Standard TESS Sector duration
        flux = np.ones_like(time)
        
        # Inject a fake planet transit dip for orchestration testing
        period = 4.2
        t0 = 1.5
        duration = 0.15
        depth = 0.015  # 1.5% transit depth
        
        phases = ((time - t0) % period) / period
        phases = np.where(phases > 0.5, phases - 1.0, phases)
        transit_mask = np.abs(phases * period) < (duration / 2)
        flux[transit_mask] = 1.0 - depth
        
        # Add slight white noise
        flux += np.random.normal(0, 0.001, size=len(time))
        
        return LightCurveData(time, flux)
