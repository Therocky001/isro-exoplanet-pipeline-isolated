from pipeline.catalog_parser import CatalogDataEngine
from pipeline.data_models import PipelineResult
from pipeline.pipeline import ExoplanetPipeline

def run_target_pipeline(
    target_id: str,
    filtered_index_path: str = "output/filtered_index.csv",
    checkpoint_path: str = "output/exoplanet_classifier.pt",
) -> PipelineResult:
    """
    End-to-end execution flow for a single target TIC ID.
    """
    print("\n" + "=" * 60)
    print(f"📡 Processing Target: {target_id}")
    print("=" * 60)
    
    # 1. DuckDB Metadata Extraction
    db_engine = CatalogDataEngine()
    try:
        stellar_meta = db_engine.get_stellar_metadata(target_id, filtered_index_path)
        print(f"✅ DuckDB Target Match: Rad={stellar_meta.stellar_radius} R_sun, Mass={stellar_meta.stellar_mass} M_sun, logg={stellar_meta.logg}")
    except Exception as e:
        print(f"❌ Metadata Extraction Failed: {str(e)}")
        return PipelineResult(status="pipeline_error")

    print("🧠 Running shared ExoplanetPipeline orchestration...")
    pipeline = ExoplanetPipeline(model_checkpoint=checkpoint_path, output_dir="output")
    result = pipeline.run(target_id)

    if result.status != "pipeline_error" and result.transit_params is not None:
        print(f"🎯 Neural Classification: {result.status.upper()} (Confidence: {max(result.class_probs.values()) * 100:.1f}%)")
        print(f"🔭 BLS Significance: {result.bls_snr:.2f} σ")

    return result

if __name__ == "__main__":
    test_target = "TIC 80423805"
    result = run_target_pipeline(test_target)
    print(f"\n🎯 Pipeline Execution Result: STATUS = {result.status} | BLS_SNR = {result.bls_snr}")