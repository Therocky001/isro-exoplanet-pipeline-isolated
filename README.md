# Exoplanet Hackathon Pipeline

Team Nakshathra's end-to-end pipeline for AI-enabled exoplanet detection from noisy TESS light curves.

## What is included
- DuckDB catalog filtering for the raw TESS TIC crossmatch table.
- TOI-based label bootstrapping from the public NASA Exoplanet Archive.
- A dual-headed 1D CNN classifier in PyTorch.
- BLS periodogram significance and transit-parameter estimation.
- Streamlit dashboard for local review and demo use.

## Quick start
1. Create and activate the provided virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Rebuild the filtered catalog index:

```bash
python run_data_prep.py
```

4. Build the fast labeled training manifest:

```bash
python -c "from bootstrap_toi_labels import build_fast_bootstrap_manifest; build_fast_bootstrap_manifest()"
```

5. Train the model:

```bash
python train_model.py --epochs 1 --batch-size 3 --manifest output/toi_labeled_training_manifest_fast.csv --checkpoint output/exoplanet_classifier.pt
```

6. Run the local Streamlit server:

```bash
streamlit run app.py --server.headless true --server.port 8501
```

## Notes
- The raw file in [data/tic_ctl_crossmatched.csv](data/tic_ctl_crossmatched.csv) is a metadata catalog, not a supervised label table.
- Public TOI dispositions from the NASA Exoplanet Archive are used to bootstrap labels for `planet`, `eclipsing_binary`, and `blend_noise`.
- The fast training path uses a small balanced bootstrap so the model can be trained locally in a practical amount of time.
- The trained checkpoint is saved at [output/exoplanet_classifier.pt](output/exoplanet_classifier.pt).

## Deployment
For GitHub deployment, commit the code, the README, and the trained checkpoint if you want the demo to run immediately after clone. If you prefer a lighter repository, exclude the cache directory and regenerate the checkpoint from the instructions above.
