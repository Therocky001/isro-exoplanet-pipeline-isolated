import numpy as np
from astropy.timeseries import BoxLeastSquares

class BLSResult:
    def __init__(self, periods, powers, raw_results):
        self.periods = periods
        self.powers = powers
        self.results = raw_results  # Safely bind astropy object
        self.reliable = True

class TransitCandidate:
    def __init__(self, period, t0, duration, depth, bls_power, n_transits):
        self.period = period
        self.t0 = t0
        self.duration = duration
        self.depth = depth
        self.bls_power = bls_power
        self.n_transits = n_transits
        self.is_eb_indicator = False

class BLSPeriodogramModule:
    def compute_periodogram(self, lc, period_min=0.5, period_max=20.0, n_periods=2000) -> BLSResult:
        periods = np.logspace(np.log10(period_min), np.log10(period_max), n_periods)
        durations = np.linspace(0.05, 0.25, 5)

        # Use explicitly extracted underlying arrays to avoid wrapper class proxy loops
        bls = BoxLeastSquares(lc.time, lc.flux)
        results = bls.power(periods, durations, objective='snr')

        bls_res = BLSResult(periods, results.power, results)
        if (lc.time.max() - lc.time.min()) < 3 * periods[np.argmax(results.power)]:
            bls_res.reliable = False

        return bls_res

    def extract_best_candidate(self, result: BLSResult) -> TransitCandidate:
        best_index = np.argmax(result.powers)

        period = float(result.periods[best_index])
        duration = float(result.results.duration[best_index])
        t0 = float(result.results.transit_time[best_index])
        depth = float(result.results.depth[best_index])
        bls_power = float(result.powers[best_index])
        
        try:
            n_transits = int(result.results.n_transits[best_index])
        except:
            n_transits = 3

        return TransitCandidate(period, t0, duration, depth, bls_power, n_transits)

    def compute_snr(self, lc, candidate: TransitCandidate) -> float:
        if candidate.period <= 0 or candidate.duration <= 0:
            return 0.0

        phase = ((lc.time - candidate.t0) % candidate.period) / candidate.period
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        in_transit = np.abs(phase) < (candidate.duration / (2.0 * candidate.period))

        out_of_transit_flux = lc.flux[~in_transit]
        if out_of_transit_flux.size < 2:
            return 0.0

        baseline = np.median(out_of_transit_flux)
        scatter = 1.4826 * np.median(np.abs(out_of_transit_flux - baseline))
        if scatter <= 0:
            scatter = np.std(out_of_transit_flux)
        if scatter <= 0:
            return 0.0

        depth_estimate = baseline - np.median(lc.flux[in_transit]) if np.any(in_transit) else candidate.depth
        n_in_transit = max(int(np.sum(in_transit)), 1)
        snr = abs(depth_estimate) / scatter * np.sqrt(n_in_transit)
        return float(np.round(snr, 2))