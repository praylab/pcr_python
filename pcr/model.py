# PCR shoreline evolution model, wrapping the stage functions in scripts/
import time

import numpy as np

from pcr import storm, slr, shoreline, erosion


class PCRModel:
    '''
    Probabilistic Coastline Recession (PCR) model.

    Wraps the storm/slr/erosion/shoreline stage functions in scripts/ into a
    single object holding simulation config and intermediate state. Stages
    are run in order via run(), or individually for inspection/testing.
    '''

    def __init__(
        self,
        year_start: str,
        year_end: str,
        nr_simulation: int = 1000,
        nr_batch: int = 1000,
        # SLR
        scenario: str = 'ssp126',
        wl0: float = 0.0,
        # erosion
        rec_rate: float = 7.4119/365, # calibrated value
        m: float = 0.024,
        c1: float = 1.339,
        c2: float = 1.983,
        # storm detection
        ts_hs: float = 95,
        ts_dur: float = 12.0,
        ts_between: float = 48.0,
        ts_hs_type: str = 'percentile', 
        # post-process
        statistics_kind: str = 'min',
        # future condition
        fac_lambda: np.array = np.array([[2015, 2100], [1, 1]]),
        rec_rate_end: float = None
    ):
        self.date_start = np.datetime64(f'{year_start}-01-01T00:00:00')
        self.date_end = np.datetime64(f'{year_end}-12-31T23:59:00')
        self.year_start = int(year_start)
        self.year_end = int(year_end)
        self.nr_simulation = nr_simulation
        self.nr_batch = nr_batch

        self.wl0 = wl0
        self.ar6_scenario = scenario

        self.rec_rate = rec_rate
        self.m = m
        self.c1 = c1
        self.c2 = c2

        self.ts_hs = ts_hs
        self.ts_dur = ts_dur
        self.ts_between = ts_between
        self.ts_hs_type = ts_hs_type

        self.statistics_kind = statistics_kind

        self.fac_lambda = fac_lambda
        self.rec_rate_end = rec_rate_end

        # populated via attach_slr()/attach_wave_data() (see pcr.builder for the
        # I/O that produces these values)
        self.rate_ar6 = None
        self.days_ar6 = None

        self.hs = None
        self.dir = None
        self.tp = None
        self.day = None
        self.record_years = None
        # the ERA5 grid point the wave data came from; used as the cache key
        # for rec_rate calibration (see pcr.calibration/pcr.io lookup table)
        self.lon_wave = None
        self.lat_wave = None

        self.transect = None

        self.detected_storm = None
        self.storm_props = None
        self.fitted_lambdas = None
        self.yearly_storm = None

        self.t_days = None
        self.t_years = None
        self.day_to_month = None
        self.day_to_year = None
        self.day_to_fac = None
        self.day_to_rec_rate = None

        self.shoreline_stats = None
        # each simulation yields a variable-length array (2 entries per storm event),
        # so these hold one array object per simulation rather than a fixed-width array
        self.track_time = [None] * nr_simulation
        self.track_shoreline = [None] * nr_simulation

    def attach_slr(self, rate_ar6, days_ar6, scenario=None):
        '''
        Attach an AR6 sea level rate curve loaded by pcr.builder.build_slr_curve().
        Exposed as a setter (rather than folded into __init__) so calibration workflows
        can swap in a freshly-loaded curve (e.g. after changing scenario) and rerun
        run_simulation() without reconstructing the model.

        :param scenario: AR6 scenario the curve was built for (e.g. 'ssp126', or '0' for
            no SLR). Pass this whenever you swap in a curve for a different scenario than
            the model was constructed with — it updates self.ar6_scenario, which is what
            actually gates vector_simulate_slrAR6 (rate_ar6/days_ar6 alone are not enough:
            scenario == '0' short-circuits to zero SLR regardless of the attached curve).
        '''
        self.rate_ar6 = rate_ar6
        self.days_ar6 = days_ar6
        if scenario is not None:
            self.ar6_scenario = scenario
        return self.rate_ar6, self.days_ar6

    def attach_wave_data(self, hs, dir, tp, day, record_years, lon_wave=None, lat_wave=None):
        '''
        Attach a wave time series loaded by pcr.builder.load_wave_data(), plus the
        number of years the record spans (used to scale yearly storm counts).

        :param lon_wave, lat_wave: coordinates of the wave data point (e.g. the
            nearest ERA5 grid point), when known. Used as the lookup key for
            cached rec_rate calibration (see pcr.calibration).
        '''
        self.hs = hs
        self.dir = dir
        self.tp = tp
        self.day = day
        self.record_years = record_years
        self.lon_wave = lon_wave
        self.lat_wave = lat_wave
        return self.hs, self.dir, self.tp, self.day

    def attach_transect(self, transect): 
        self.transect = transect
        return self.transect

    def detect_storms(self):
        '''Detect storms from the attached wave data and fit storm/gap distributions.'''
        if self.hs is None:
            raise RuntimeError('call attach_wave_data() before detect_storms()')

        self.detected_storm, _ = storm.detect(
            self.hs, self.dir, self.tp, self.day, self.ts_hs, self.ts_dur, self.ts_between
        )
        self.storm_props = storm.fit_storm(self.detected_storm)
        self.fitted_lambdas = storm.fit_lambda_gap(self.detected_storm)
        self.yearly_storm = np.ceil(self.detected_storm.shape[0] / self.record_years)

        return self.detected_storm, self.storm_props, self.fitted_lambdas

    def prepare_simulation(self):
        '''Precompute simulation horizon, day-to-month mapping, and the result array.'''
        self.t_days = (self.date_end - self.date_start).item().days
        self.t_years = self.year_end - self.year_start

        self.day_to_month, self.day_to_year = storm.build_day_to_month_year(self.date_start, self.t_days)
        # fac_lambda and day_to_year are fixed for the whole run, so interpolate the
        # lambda-scaling factor once here rather than per proposed storm arrival
        # inside gap_nhpp_thinning's hot loop
        self.day_to_fac = np.interp(self.day_to_year, self.fac_lambda[0], self.fac_lambda[1])

        # recovery rate ramps linearly from rec_rate (at year_start) to rec_rate_end
        # (at year_end); if rec_rate_end isn't given, this collapses to the constant
        # rec_rate everywhere. Built once here as a day-indexed lookup, same pattern
        # as day_to_fac, rather than interpolated per simulation.
        end_rate = self.rec_rate if self.rec_rate_end is None else self.rec_rate_end
        self.day_to_rec_rate = np.interp(self.day_to_year, [self.year_start, self.year_end], [self.rec_rate, end_rate])

        self.shoreline_stats = np.empty((self.t_years + 1, self.nr_simulation))
        self.track_time = [None] * self.nr_simulation
        self.track_shoreline = [None] * self.nr_simulation

        return self.shoreline_stats

    def compute_statistics(self, nr_sim):
        '''compute annual statistics'''
        row = shoreline.vector_get_annual_statistics(
                track_time=self.track_time[nr_sim],
                shoreline_position=self.track_shoreline[nr_sim],
                kind=self.statistics_kind,
                date_start=self.date_start,
                date_end=self.date_end,
                year_start=self.year_start,
                year_end=self.year_end,
            )

        return row

    def _generate_batch(self):
        '''Sample a batch of synthetic storms sized for the full simulation horizon.'''
        n_sample = self.yearly_storm * self.t_years * self.nr_batch
        return storm.generate(
            fitted_storm=self.storm_props,
            sampling_size=n_sample,
            oversample=0.1,
            max_dur=np.max(self.detected_storm.duration),
        )

    def _simulate_one(self, hss, durs, dirs, tps, storm_count):
        '''Run a single simulation starting at storm_count into the sampled batch arrays.'''
        synth_start = storm.gap_nhpp_thinning(
            T=self.t_days,
            monthly_lambda=self.fitted_lambdas,
            date_start=self.date_start,
            duration=durs,
            start_storm=storm_count,
            day_to_month=self.day_to_month,
            day_to_year=self.day_to_year,
            day_to_fac=self.day_to_fac, 
            fac_lambda=self.fac_lambda
        )

        storm_count_end = storm_count + len(synth_start)

        synth_hs, synth_direction, synth_duration, synth_tp, synth_end, synth_gap = storm.slice_synthetic_batch(
            hss, dirs, durs, tps, synth_start, storm_count, storm_count_end
        )

        synth_slr = slr.vector_simulate_slrAR6(
            day_start=synth_start,
            rate_sl=self.rate_ar6,
            day_sl=self.days_ar6,
            wl0=self.wl0,
            scenario=self.ar6_scenario,
        )

        _, synth_erosion = erosion.vector_mendoza(
            hss=synth_hs,
            tps=synth_tp,
            durs=synth_duration,
            m=self.m,
            C1=self.c1,
            C2=self.c2,
        )

        # look up each storm's recovery rate for its day in one vectorized indexing
        # op (day_to_rec_rate is precomputed once in prepare_simulation)
        day_idx = np.clip(synth_start.astype(int), 0, len(self.day_to_rec_rate) - 1)
        synth_recovery = shoreline.vector_calculate_recovery(
            gaps=synth_gap,
            rec_rate=self.day_to_rec_rate[day_idx],
        )

        synth_retreat = shoreline.vector_calculate_slr_retreat(
            slrs=synth_slr,
            m=self.m,
        )

        track_time, _, track_shoreline_position = shoreline.vector_track_shoreline(
            day_start=synth_start,
            day_end=synth_end,
            recovery=synth_recovery,
            retreat=synth_retreat,
            erosion=synth_erosion,
        )

        return track_time, track_shoreline_position, storm_count_end

    def run_simulation(self):
        '''Run the batched Monte-Carlo simulation stage (assumes attach_slr/attach_wave_data/detect_storms already ran).'''
        self.prepare_simulation()

        sim_count = 0
        while sim_count < self.nr_simulation:
            hss, durs, dirs, tps = self._generate_batch()
            storm_count = 0

            print(f'progress: {sim_count / self.nr_simulation * 100:.2f} %')
            for _ in range(self.nr_batch):
                self.track_time[sim_count], self.track_shoreline[sim_count], storm_count = self._simulate_one(hss, durs, dirs, tps, storm_count)

                row = self.compute_statistics(sim_count)
                self.shoreline_stats[:, sim_count] = row.flatten()

                sim_count += 1
                if sim_count >= self.nr_simulation:
                    break

        # return self.shoreline_stats

    def run(self):
        '''
        Run the simulation pipeline: storm detection then batched simulation.
        Assumes the SLR curve and wave data have already been loaded and attached
        (see pcr.builder.build_model, or attach_slr()/attach_wave_data() directly).
        '''
        t0 = time.time()

        print('Detecting storms ...')
        self.detect_storms()

        print('Starting the simulation ...')
        self.run_simulation()

        print(f'running for {time.time() - t0:2f} s')
        # return self.shoreline_stats, error_count
