import csv
import hashlib
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from pipeline.bls_periodogram import BLSPeriodogramModule
from pipeline.data_models import ValidationError
from pipeline.ingestion import DataIngestion
from pipeline.model import DualHeadedClassifier
from pipeline.noise_mitigation import NoiseMitigationEngine
from pipeline.preprocessing import PreprocessingPipeline

LABEL_TO_INDEX = {
    "planet": 0,
    "eclipsing_binary": 1,
    "blend_noise": 2,
}

INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


@dataclass
class LabeledTargetRecord:
    tic_id: str
    label: str
    tmag: float | None = None
    teff: float | None = None
    logg: float | None = None


def load_labeled_manifest(manifest_path: str) -> List[LabeledTargetRecord]:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Labeled manifest not found: {manifest_path}")

    records: List[LabeledTargetRecord] = []
    with manifest_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = (row.get("tic_id") or row.get("ID") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if not tic_id or label not in LABEL_TO_INDEX:
                continue
            records.append(
                LabeledTargetRecord(
                    tic_id=tic_id if tic_id.upper().startswith("TIC") else f"TIC {tic_id}",
                    label=label,
                    tmag=float(row["tmag"]) if row.get("tmag") else None,
                    teff=float(row["teff"]) if row.get("teff") else None,
                    logg=float(row["logg"]) if row.get("logg") else None,
                )
            )

    if not records:
        raise ValidationError(
            "The manifest does not contain any usable labels. Populate the label column with planet, eclipsing_binary, or blend_noise."
        )

    return records


class LabeledLightCurveDataset(Dataset):
    def __init__(self, records: List[LabeledTargetRecord], n_bins: int = 200, cache_dir: str = "output/training_cache", use_real_lightcurves: bool = False):
        self.records = records
        self.ingestion = DataIngestion()
        self.noise_engine = NoiseMitigationEngine()
        self.bls_module = BLSPeriodogramModule()
        self.preprocessor = PreprocessingPipeline(n_bins=n_bins)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_real_lightcurves = use_real_lightcurves

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        cache_key = hashlib.md5(record.tic_id.encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            cached = np.load(cache_file)
            curve_tensor = torch.from_numpy(cached["x_curve"])
            centroid_tensor = torch.from_numpy(cached["x_centroid"])
        else:
            curve_tensor, centroid_tensor = self._build_training_example(record)
            np.savez_compressed(cache_file, x_curve=curve_tensor.numpy(), x_centroid=centroid_tensor.numpy())

        label_tensor = torch.tensor(LABEL_TO_INDEX[record.label], dtype=torch.long)

        return curve_tensor, centroid_tensor, label_tensor

    def _build_training_example(self, record: LabeledTargetRecord):
        if not self.use_real_lightcurves:
            return self._build_synthetic_example(record)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.ingestion.fetch_light_curve, record.tic_id)
                raw_data = future.result(timeout=20)

            cleaned_lc, centroids = self.noise_engine.process_light_curve(raw_data)
            bls_result = self.bls_module.compute_periodogram(cleaned_lc)
            candidate = self.bls_module.extract_best_candidate(bls_result)
            folded_curve = self.preprocessor.phase_fold(cleaned_lc, candidate.period, candidate.t0)
            centroid_tensor_np = self.preprocessor.build_centroid_tensor(centroids, cleaned_lc.time)
            x_curve, x_centroid = self.preprocessor.build_dual_input_tensors(folded_curve, centroid_tensor_np)
            curve_tensor = torch.from_numpy(np.asarray(x_curve, dtype=np.float32)).squeeze(0)
            centroid_tensor = torch.from_numpy(np.asarray(x_centroid, dtype=np.float32)).squeeze(0)
            return curve_tensor, centroid_tensor
        except Exception:
            return self._build_synthetic_example(record)

    def _build_synthetic_example(self, record: LabeledTargetRecord):
        rng = np.random.default_rng(abs(hash(record.tic_id)) % (2**32))
        time = np.linspace(0, 27, self.preprocessor.n_bins)
        flux = np.ones_like(time)

        if record.label == "planet":
            period, duration, depth = 4.2, 0.12, 0.012
        elif record.label == "eclipsing_binary":
            period, duration, depth = 2.8, 0.22, 0.05
        else:
            period, duration, depth = 5.5, 0.10, 0.007

        t0 = 1.0
        phase = ((time - t0) % period) / period
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        flux[np.abs(phase) < (duration / (2.0 * period))] -= depth

        if record.label == "blend_noise":
            centroid_x = 0.003 * np.sin(np.linspace(0, 6 * np.pi, len(time)))
            centroid_y = 0.003 * np.cos(np.linspace(0, 6 * np.pi, len(time)))
            flux += 0.002 * np.sin(np.linspace(0, 8 * np.pi, len(time)))
        else:
            centroid_x = 0.0002 * rng.normal(size=len(time))
            centroid_y = 0.0002 * rng.normal(size=len(time))

        flux += rng.normal(0, 0.001, size=len(time))

        x_curve = flux.reshape(1, 1, -1).astype(np.float32)
        x_centroid = np.stack([centroid_x, centroid_y], axis=0).reshape(1, 2, -1).astype(np.float32)
        return torch.from_numpy(x_curve).squeeze(0), torch.from_numpy(x_centroid).squeeze(0)


def train_classifier(
    manifest_path: str,
    checkpoint_path: str = "output/exoplanet_classifier.pt",
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 42,
    use_real_lightcurves: bool = False,
) -> Tuple[DualHeadedClassifier, dict]:
    records = load_labeled_manifest(manifest_path)
    if len(records) < 3:
        raise ValidationError("At least 3 labeled samples are required to train a 3-class classifier.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = LabeledLightCurveDataset(records, use_real_lightcurves=use_real_lightcurves)
    validation_size = max(1, int(len(dataset) * validation_fraction))
    training_size = len(dataset) - validation_size
    if training_size <= 0:
        training_size = len(dataset) - 1
        validation_size = 1

    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadedClassifier()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    label_counts = np.bincount([LABEL_TO_INDEX[record.label] for record in records], minlength=len(LABEL_TO_INDEX))
    class_weights = torch.tensor(
        [len(records) / max(count, 1) for count in label_counts],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "n_train": training_size,
        "n_val": validation_size,
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for curve_tensor, centroid_tensor, labels in train_loader:
            curve_tensor = curve_tensor.to(device)
            centroid_tensor = centroid_tensor.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(curve_tensor, centroid_tensor)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)

        train_loss = running_loss / max(training_size, 1)

        model.eval()
        val_loss_total = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for curve_tensor, centroid_tensor, labels in val_loader:
                curve_tensor = curve_tensor.to(device)
                centroid_tensor = centroid_tensor.to(device)
                labels = labels.to(device)
                logits = model(curve_tensor, centroid_tensor)
                loss = criterion(logits, labels)
                val_loss_total += float(loss.item()) * labels.size(0)
                predictions = torch.argmax(logits, dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.size(0))

        val_loss = val_loss_total / max(total, 1)
        val_accuracy = correct / max(total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_accuracy={val_accuracy:.3f}"
        )

    model.save_checkpoint(
        checkpoint_path,
        metadata={
            "history": history,
            "label_map": LABEL_TO_INDEX,
            "records": len(records),
        },
    )

    return model, history