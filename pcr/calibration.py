# Recovery-rate calibration: fit rec_rate so the median simulated shoreline
# stays flat under zero SLR, then cache the result in a lookup table keyed by
# wave point (see pcr.io load_rec_rate_lookup/save_rec_rate_lookup) so repeat
# runs at the same location don't have to re-run the (expensive) fit.
import numpy as np
from scipy.optimize import minimize_scalar

from pcr import io

DEFAULT_GRID_DAYS = np.arange(1, 366 * 100, 30)


def calibrate_rec_rate(model, bounds=(1/365, 10/365), grid_days=DEFAULT_GRID_DAYS, seed=0, xatol=1e-6):
    '''
    Calibrate model.rec_rate so that, under zero sea level rise, the median
    shoreline position across the simulated horizon stays as close to flat
    (zero net change) as possible. Mutates model.rec_rate to the fitted value.

    Runs model.run_simulation() once per candidate rec_rate with a fixed random
    seed (so every candidate sees the same synthetic storm sequence), which makes
    this slow — prefer get_or_calibrate_rec_rate() to reuse a cached result for
    repeat runs at the same wave point.

    :return: the scipy.optimize OptimizeResult from the search.
    '''
    # attach 0 sea level change for calibration
    model.attach_slr([0, 0], [0, 0], scenario='0')

    def evaluate_rec_rate(rec_rate):
        model.rec_rate = rec_rate

        np.random.seed(seed)  # same storm sequence for every candidate rec_rate
        model.run_simulation()

        aligned = np.empty((len(grid_days), len(model.track_time)))
        for i, (t, s) in enumerate(zip(model.track_time, model.track_shoreline)):
            aligned[:, i] = np.interp(grid_days, t, s)

        median_all = np.median(aligned, axis=1)
        return np.sum(np.abs(median_all))

    # calibration needs the per-simulation tracks regardless of what the model
    # was built with; force it on here and restore the caller's setting after,
    # even if the search below raises
    keep_tracks = model.keep_tracks
    model.keep_tracks = True
    try:
        result = minimize_scalar(
            evaluate_rec_rate,
            bounds=bounds,
            method='bounded',
            options={'xatol': xatol},
        )
    finally:
        model.keep_tracks = keep_tracks

    model.rec_rate = result.x
    return result


def get_or_calibrate_rec_rate(model, force=False, tol=1e-6, **calibrate_kwargs):
    '''
    Reuse a cached rec_rate calibration for model's wave point (model.lon_wave/
    lat_wave, attached by pcr.builder.build_model) if one exists; otherwise run
    calibrate_rec_rate() and record the result for next time. Mutates and
    returns model.rec_rate.

    :param force: recalibrate even if a cached value exists, overwriting it.
    :param tol: degrees within which an existing table entry counts as "the same"
        wave point (see pcr.io.find_rec_rate/save_rec_rate_lookup).
    :param calibrate_kwargs: forwarded to calibrate_rec_rate() (grid_days, seed,
        bounds, xatol) when a fresh calibration is needed.
    '''
    if model.lon_wave is None or model.lat_wave is None:
        raise ValueError(
            'model.lon_wave/lat_wave are not set -- attach wave data with a '
            'location (e.g. via pcr.builder.build_model with source="ARCO") '
            'before calibrating, or call calibrate_rec_rate() directly to skip '
            'the lookup table.'
        )

    if not force:
        cached = io.find_rec_rate(model.lon_wave, model.lat_wave, tol=tol)
        if cached is not None:
            print(f'found cached rec_rate for ({model.lon_wave:.4f}, {model.lat_wave:.4f}): '
                  f'{cached:.6f} (x*365 = {cached*365:.4f} m/year)')
            model.rec_rate = cached
            return model.rec_rate

    result = calibrate_rec_rate(model, **calibrate_kwargs)
    io.save_rec_rate_lookup(model.lon_wave, model.lat_wave, result.x, objective=result.fun, tol=tol)

    print(f'calibrated rec_rate for ({model.lon_wave:.4f}, {model.lat_wave:.4f}): '
          f'{result.x:.6f} (x*365 = {result.x*365:.4f} m/year), objective: {result.fun:.4f}')

    return model.rec_rate
