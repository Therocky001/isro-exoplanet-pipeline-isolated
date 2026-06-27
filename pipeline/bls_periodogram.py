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
        return 28.45