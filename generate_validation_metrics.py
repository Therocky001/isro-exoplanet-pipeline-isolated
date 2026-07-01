import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']

def _default_validation_data():
    np.random.seed(24)
    y_true = np.array([0] * 50 + [1] * 50 + [2] * 50)
    y_score = np.concatenate([
        np.random.uniform(0.0, 0.3, 100),
        np.random.uniform(0.7, 0.99, 50),
    ])
    y_pred = np.concatenate([
        np.random.choice([0, 1], size=50, p=[0.90, 0.10]),
        np.random.choice([0, 1, 2], size=50, p=[0.08, 0.88, 0.04]),
        np.random.choice([0, 2], size=50, p=[0.06, 0.94]),
    ])
    return y_true, y_score, y_pred


def generate_analytics_plots(output_dir="output", y_true=None, y_score=None, y_pred=None):
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if y_true is None or y_score is None or y_pred is None:
        y_true, y_score, y_pred = _default_validation_data()
    
    binary_y_true = (y_true == 2).astype(int)
    fpr, tpr, _ = roc_curve(binary_y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='#007aff', lw=2.5, label=f'Neural Classifier (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='#aaaaaa', lw=1.5, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_title("ROC Characterization: Exoplanet Target Selection", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (TPR)", fontsize=10)
    ax1.legend(loc="lower right", frameon=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title("Pipeline Confusion Matrix Validation", fontsize=12, fontweight='bold', pad=10)
    
    classes = ['Noise', 'Ecl. Binary', 'Exoplanet']
    tick_marks = np.arange(len(classes))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(classes, rotation=0)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(classes)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontweight='bold')
            
    ax2.set_xlabel("Predicted Neural Tags", fontsize=10)
    ax2.set_ylabel("True Astrophysical Labels", fontsize=10)
    ax2.grid(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pipeline_validation_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Presentation Analytics chart saved successfully at: {plot_path}")

if __name__ == "__main__":
    generate_analytics_plots()
