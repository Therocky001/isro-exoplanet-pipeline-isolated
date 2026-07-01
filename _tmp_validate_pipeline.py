from run_pipeline import run_target_pipeline

result = run_target_pipeline('TIC 80423805')
print(result.status)
print(result.bls_snr)
print(result.class_probs)
