import numpy as np

class DataValidator:
    @staticmethod
    def validate_light_curve(time: np.ndarray, flux: np.ndarray) -> bool:
        if time is None or flux is None:
            return False
        if len(time) != len(flux):
            return False
        if len(time) < 100:  # Minimum cadences required
            return False
        if np.any(np.isnan(time)):
            return False
        return True
