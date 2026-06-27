import numpy as np
from typing import Dict, Any
from pipeline.data_models import TransitParameters

class TransitPhysicsModeler:
    def __init__(self, r_star: float = 1.0, m_star: float = 1.0):
        """
        r_star: Radius of the host star in solar radii (R_sun)
        m_star: Mass of the host star in solar masses (M_sun)
        """
        self.r_star = r_star
        self.m_star = m_star
        
        # Physical constants in SI units
        self.G = 6.67430e-11
        self.R_sun = 6.95700e8  # meters
        self.M_sun = 1.98847e30  # kg
        self.R_jup = 7.1492e7    # meters
        self.R_earth = 6.371e6   # meters

    def characterization(self, target_id: str, period: float, depth: float, duration: float) -> TransitParameters:
        """
        Derives physical parameters using Keplerian laws and basic transit geometry.
        """
        # 1. Calculate Semi-major axis 'a' using Kepler's Third Law: a^3 = G * M * P^2 / (4 * pi^2)
        period_seconds = period * 86400.0
        m_star_kg = self.m_star * self.M_sun
        
        a_cubed = (self.G * m_star_kg * (period_seconds ** 2)) / (4 * (np.pi ** 2))
        a_meters = a_cubed ** (1.0 / 3.0)
        a_au = a_meters / 1.495978707e11  # Convert to Astronomical Units (AU)

        # 2. Calculate Planet Radius (R_p / R_s = sqrt(depth))
        # depth here is fractional (e.g., 0.01 for 1%)
        rp_rs = np.sqrt(depth)
        r_planet_meters = rp_rs * (self.r_star * self.R_sun)
        
        r_earth = r_planet_meters / self.R_earth
        r_jup = r_planet_meters / self.R_jup

        # Pack derived values into the standard dataclass structure
        return TransitParameters(
            target_id=target_id,
            period=period,
            period_err=period * 0.001,
            t0=0.0,
            t0_err=0.0,
            rp_rs=rp_rs,
            rp_rs_err=rp_rs * 0.02,
            transit_depth=depth,
            transit_depth_err=depth * 0.01,
            duration=duration,
            duration_err=duration * 0.05,
            impact_parameter=0.2,  # Baseline approximation
            impact_parameter_err=0.05,
            r_hat={"radius_earth": r_earth, "radius_jupiter": r_jup, "semi_major_axis_au": a_au}
        )
