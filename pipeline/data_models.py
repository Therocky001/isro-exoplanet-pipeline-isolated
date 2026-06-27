from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class TransitParameters:
    target_id: str
    period: float
    period_err: float
    t0: float
    t0_err: float
    rp_rs: float
    rp_rs_err: float
    transit_depth: float
    transit_depth_err: float
    duration: float
    duration_err: float
    impact_parameter: float
    impact_parameter_err: float
    r_hat: Dict[str, float] = field(default_factory=dict)

@dataclass
class PipelineResult:
    status: str  # 'planet', 'eclipsing_binary', 'blend_noise', 'insufficient_data', 'no_signal', 'pipeline_error'
    transit_params: Optional[TransitParameters] = None
    bls_snr: float = 0.0
