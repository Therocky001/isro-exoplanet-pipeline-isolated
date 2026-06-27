import logging
import concurrent.futures
from typing import List, Dict, Any, Optional
from pipeline.data_models import PipelineResult, TransitParameters
from pipeline.ingestion import DataIngestion
from pipeline.noise_mitigation import NoiseMitigationEngine
from pipeline.bls_periodogram import BLSPeriodogramModule
from pipeline.preprocessing import PreprocessingPipeline
from pipeline.model import DualHeadedClassifier
from pipeline.transit_modeler import TransitPhysicsModeler

logger = logging.getLogger("ExoplanetPipeline")

class ExoplanetPipeline:
    def __init__(self, model_checkpoint: str, output_dir: str = "output/", snr_threshold: float = 7.0):
        self.output_dir = output_dir
        self.snr_threshold = snr_threshold
        
        # Instantiate modules
        self.ingestion = DataIngestion()
        self.noise_engine = NoiseMitigationEngine()
        self.bls_module = BLSPeriodogramModule()
        self.prep_pipeline = PreprocessingPipeline()
        self.classifier = DualHeadedClassifier()
        self.physics_engine = TransitPhysicsModeler(r_star=1.0, m_star=1.0) # Assume solar-type star baseline
        
        logger.info("Pipeline modules initialized successfully.")

    def run(self, target_id: str, sectors: List[int] = None, cadence: str = "short") -> PipelineResult:
        try:
            raw_data = self.ingestion.fetch_light_curve(target_id, sectors, cadence)
            cleaned_lc, centroids = self.noise_engine.process_light_curve(raw_data)
            
            if cleaned_lc is None:
                return PipelineResult(status="insufficient_data")
                
            bls_result = self.bls_module.compute_periodogram(cleaned_lc)
            candidate = self.bls_module.extract_best_candidate(bls_result)
            
            snr = self.bls_module.compute_snr(cleaned_lc, candidate)
            if snr < self.snr_threshold:
                return PipelineResult(status="no_signal", bls_snr=snr)
                
            # Preprocessing for AI Inference
            folded_curve = self.prep_pipeline.phase_fold(cleaned_lc, candidate.period, candidate.t0)
            centroid_tensor_np = self.prep_pipeline.build_centroid_tensor(centroids, cleaned_lc.time)
            x_curve, x_centroid = self.prep_pipeline.build_dual_input_tensors(folded_curve, centroid_tensor_np)
            
            prob_dict = self.classifier.predict_proba(x_curve, x_centroid)
            best_label = max(prob_dict, key=prob_dict.get)
            
            if best_label == "planet":
                # Trigger live Physics characterization engine instead of static placeholders
                params = self.physics_engine.characterization(
                    target_id=target_id,
                    period=candidate.period,
                    depth=candidate.depth,
                    duration=candidate.duration
                )
                return PipelineResult(status="planet", transit_params=params, bls_snr=snr)
            elif best_label == "eclipsing_binary":
                return PipelineResult(status="eclipsing_binary", bls_snr=snr)
            else:
                return PipelineResult(status="blend_noise", bls_snr=snr)
                
        except Exception as e:
            logger.error(f"Pipeline error on target {target_id}: {str(e)}")
            return PipelineResult(status="pipeline_error")

    def run_batch(self, targets_list: List[Dict[str, Any]], max_workers: int = 8) -> List[PipelineResult]:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run, t["target_id"], t.get("sectors"), t.get("cadence", "short")): t 
                for t in targets_list
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return results