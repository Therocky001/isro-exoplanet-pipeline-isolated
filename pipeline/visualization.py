import os
import matplotlib.pyplot as plt
import numpy as np

class VisualizationEngine:
    def __init__(self, output_dir: str = "output"):
        """Initializes the engine and ensures the output directory exists."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Apple-style clean minimal layout configurations
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
        plt.rcParams['figure.facecolor'] = '#ffffff'

    def plot_diagnostic_curves(self, tic_id: str, time: np.ndarray, raw_flux: np.ndarray, 
                               clean_flux: np.ndarray, phase: np.ndarray, folded_flux: np.ndarray):
        """
        Generates and saves a two-panel publication-grade diagnostic plot:
        1. Raw vs Cleaned Light Curve (Time-domain)
        2. Phase-Folded Transit Dip (Phase-domain centered at 0)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [1, 1]})
        fig.suptitle(f"Pipeline Diagnostic Telemetry — {tic_id}", fontsize=14, fontweight='bold', color='#111111')

        # Panel 1: Time Domain (Raw vs Mitigation Layer)
        ax1.scatter(time, raw_flux, s=2, color='#aaaaaa', alpha=0.4, label='Raw TESS Data')
        ax1.plot(time, clean_flux, color='#007aff', linewidth=1.2, label='Matrix Cleansed Stream')
        ax1.set_title("Time Domain: Noise Mitigation Verification", fontsize=11, fontweight='semibold', loc='left')
        ax1.set_xlabel("Time (BTJD)", fontsize=9, color='#333333')
        ax1.set_ylabel("Normalized Flux", fontsize=9, color='#333333')
        ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='none')
        ax1.tick_params(labelsize=8)

        # Panel 2: Phase Folded Space (Transit architecture)
        ax2.scatter(phase, folded_flux, s=3, color='#34c759', alpha=0.6, label='Binned Phase-Folded States')
        ax2.set_title("Phase Domain: Centered Transit Dip Architecture", fontsize=11, fontweight='semibold', loc='left')
        ax2.set_xlabel("Phase", fontsize=9, color='#333333')
        ax2.set_ylabel("Relative Flux", fontsize=9, color='#333333')
        ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='none')
        ax2.tick_params(labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save output file cleanly
        clean_tic_id = tic_id.replace(" ", "_")
        output_file_path = os.path.join(self.output_dir, f"{clean_tic_id}_diagnostic.png")
        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎨 Diagnostic plot generated successfully at: {output_file_path}")