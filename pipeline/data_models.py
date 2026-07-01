from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# --- Custom Exception Classes ---
class ValidationError(Exception):
    """Raised when data validations or constraints fail."""
    pass

class InsecureConnectionError(Exception):
    """Raised when network calls do not use HTTPS/TLS."""
    pass

class ModelNotFoundError(Exception):
    """Raised when the AI model checkpoint cannot be loaded."""
    pass

class PhysicalConstraintError(Exception):
    """Raised when physical constraints (like Kepler's Law bounds) are violated."""
    pass

# --- Existing Dataclasses ---
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
    status: str  # 'planet', 'eclipsing_binary', 'blend_noise', etc.
    transit_params: Optional[TransitParameters] = None
    bls_snr: float = 0.0
    class_probs: Dict[str, float] = field(default_factory=dict)

@dataclass
class StellarParameters:
    tic_id: str
    stellar_radius: float
    stellar_mass: float
    teff: float
    logg: float
    metallicity: float = 0.0