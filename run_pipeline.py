import os
import sys
import logging
from pipeline.pipeline import ExoplanetPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainRunner")

def main():
    logger.info("Initializing Team Nakshathra Automated Batch Pipeline...")
    
    pipeline = ExoplanetPipeline(model_checkpoint="checkpoints/weights.pth")
    
    batch_targets = [
        {"target_id": "TIC 261136679"},
        {"target_id": "TIC 147551061"},
        {"target_id": "TIC 278779532"}
    ]
    
    print("\n" + "="*40)
    print("      BATCH RUN EXECUTION SUMMARY      ")
    print("="*40)
    
    for target in batch_targets:
        res = pipeline.run(target["target_id"])
        
        if res.status == "planet" and res.transit_params:
            p = res.transit_params
            print(f"\n✨ Target: {target['target_id']} ---> STATUS: {res.status.upper()}")
            print(f"   • Orbital Period   : {p.period:.4f} Days")
            print(f"   • Semi-Major Axis  : {p.r_hat['semi_major_axis_au']:.4f} AU")
            print(f"   • Planet Radius    : {p.r_hat['radius_earth']:.2f} Earth Radii")
            print(f"   • Transit Depth    : {p.transit_depth * 100:.3f}%")
        else:
            print(f"❌ Target: {target['target_id']} ---> Pipeline Status: {res.status}")
            
    print("\n" + "="*40)

if __name__ == "__main__":
    main()