import numpy as np

class PreprocessingPipeline:
    def __init__(self, n_bins: int = 200):
        self.n_bins = n_bins

    def phase_fold(self, cleaned_lc, period: float, t0: float):
        time = cleaned_lc.time
        flux = cleaned_lc.flux
        
        # Calculate phases between -0.5 and 0.5
        phases = ((time - t0) % period) / period
        phases = np.where(phases > 0.5, phases - 1.0, phases)
        
        # Sort by phase
        sort_idx = np.argsort(phases)
        return {"phase": phases[sort_idx], "flux": flux[sort_idx]}

    def build_centroid_tensor(self, centroids, time):
        return centroids  # Return baseline matrix matching dims

    def build_dual_input_tensors(self, folded_curve, centroid_tensor_np):
        # Bin the data smoothly to match self.n_bins
        bins = np.linspace(-0.5, 0.5, self.n_bins + 1)
        bin_means = []
        
        for i in range(self.n_bins):
            mask = (folded_curve["phase"] >= bins[i]) & (folded_curve["phase"] < bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(folded_curve["flux"][mask]))
            else:
                bin_means.append(1.0)
                
        x_curve = np.array(bin_means)
        
        # CRITICAL FIX: Instead of throwing an error if the dip isn't in the middle,
        # we dynamically roll/shift the array so the minimum value (the transit dip)
        # lands EXACTLY at index n_bins // 2.
        min_idx = np.argmin(x_curve)
        target_midpoint = self.n_bins // 2
        shift_amount = target_midpoint - min_idx
        
        x_curve = np.roll(x_curve, shift_amount)
        
        # Mock matching dimensions for the dual-headed neural network inputs
        x_curve_tensor = np.expand_dims(np.expand_dims(x_curve, axis=0), axis=0) # Shape: (1, 1, n_bins)
        x_centroid_tensor = np.zeros((1, 2, self.n_bins))
        
        return x_curve_tensor, x_centroid_tensor