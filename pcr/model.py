# PCR shoreline evolution model, wrapping the stage functions in scripts/
import time

import numpy as np
import xarray as xr

from pcr import helper, storm, slr, shoreline, erosion


class PCRModel:
    '''
    Probabilistic Coastline Recession (PCR) model.

    Wraps the storm/slr/erosion/shoreline stage functions in scripts/ into a
    single object holding simulation config and intermediate state. Stages
    are run in order via run(), or individually for inspection/testing.
    '''

    def __init__(
        self,
        date_start: np.datetime64,
        date_end: np.datetime64,
        nr_simulation: int = 1000,
        nr_batch: int = 1000,
        # SLR
        scenario: str = 'ssp126',
        wl0: float = 0.0,
        lon_sl: float = 82,
        lat_sl: float = 7.5,
        # erosion
        doe: float = 2.5,
        ws: float = 0.04,
        d: float = 2 + 1,
        rec_rate: float = 7.5851 / 365, # calibrated value
        m: float = 0.024,
        c1: float = 1.339,
        c2: float = 1.983,
        # storm detection
        ts_hs: float = 95,
        ts_dur: float = 12.0,
        ts_between: float = 48.0,
        # wave data
        # TODO: separate the pre-process data with the model -> write in io
        lon: float = 82,
        lat: float = 7.5,
        wave_data_path: str = './data/ERA5/B3_offshore.nc',
        data_mapper: dict = None,
        # post-process
        statistics_kind: str = 'min'
    ):
        self.date_start = date_start
        self.date_end = date_end
        self.nr_simulation = nr_simulation
        self.nr_batch = nr_batch

        self.scenario = scenario
        self.wl0 = wl0
        self.ar6_scenario = scenario
        self.lon_sl = lon_sl
        self.lat_sl = lat_sl

        self.doe = doe
        self.ws = ws
        self.d = d
        self.rec_rate = rec_rate
        self.m = m
        self.c1 = c1
        self.c2 = c2

        self.ts_hs = ts_hs
        self.ts_dur = ts_dur
        self.ts_between = ts_between

        self.lon = lon
        self.lat = lat
        self.wave_data_path = wave_data_path
        self.data_mapper = data_mapper or {'hs': 'swh', 'dir': 'mwd', 'tp': 'mwp', 'time': 'time'}

        self.statistics_kind = statistics_kind

        # populated by the stage methods below
        self.rate_ar6 = None
        self.days_ar6 = None

        self.wave_data = None
        self.hs = None
        self.dir = None
        self.tp = None
        self.day = None

        self.detected_storm = None
        self.fitted_storm = None
        self.fitted_lambdas = None
        self.yearly_storm = None

        self.t_days = None
        self.t_years = None
        self.day_to_month = None

        self.shoreline_stats = None
        # each simulation yields a variable-length array (2 entries per storm event),
        # so these hold one array object per simulation rather than a fixed-width array
        self.track_time = [None] * nr_simulation
        self.track_shoreline = [None] * nr_simulation

    def init_slr(self):
        '''Import the AR6 sea level rate curve at (lon_sl, lat_sl).'''
        if self.ar6_scenario != '0':
            self.rate_ar6, self.days_ar6 = slr.import_ar6_curve(
                self.ar6_scenario, self.lon_sl, self.lat_sl, date_start=self.date_start
            )
        else: 
            self.rate_ar6, self.days_ar6 = np.zeros(10), np.zeros(10)
        return self.rate_ar6, self.days_ar6

    def load_wave_data(self):
        '''Load wave time series and extract hs, dir, tp, day arrays.'''
        self.wave_data = xr.open_dataset(self.wave_data_path)
        self.hs, self.dir, self.tp, self.day = helper.era5_input(self.wave_data, self.data_mapper)
        return self.hs, self.dir, self.tp, self.day

    def detect_storms(self):
        '''Detect storms from the loaded wave data and fit storm/gap distributions.'''
        if self.hs is None:
            raise RuntimeError('call load_wave_data() before detect_storms()')

        self.detected_storm, _ = storm.detect(
            self.hs, self.dir, self.tp, self.day, self.ts_hs, self.ts_dur, self.ts_between
        )
        self.fitted_storm = storm.fit_storm(self.detected_storm)
        self.fitted_lambdas = storm.fit_lambda_gap(self.detected_storm)

        time_var = self.data_mapper['time']
        record_years = (
            self.wave_data[time_var][-1].dt.year - self.wave_data[time_var][0].dt.year
        ).values
        self.yearly_storm = np.ceil(self.detected_storm.shape[0] / record_years)

        return self.detected_storm, self.fitted_storm, self.fitted_lambdas

    def prepare_simulation(self):
        '''Precompute simulation horizon, day-to-month mapping, and the result array.'''
        self.t_days = (self.date_end - self.date_start).item().days
        self.t_years = (
            self.date_end.astype('datetime64[Y]') - self.date_start.astype('datetime64[Y]')
        ).astype(int)

        self.day_to_month = storm.build_day_to_month(self.date_start, self.t_days)
        self.shoreline_stats = np.empty((self.t_years + 1, self.nr_simulation))

        return self.shoreline_stats

    def compute_statistics(self, nr_sim):
        '''compute annual statistics'''
        row = shoreline.vector_get_annual_statistics(
                track_time=self.track_time[nr_sim],
                shoreline_position=self.track_shoreline[nr_sim],
                kind=self.statistics_kind,
                date_start=self.date_start,
            )

        return row

    def _generate_batch(self):
        '''Sample a batch of synthetic storms sized for the full simulation horizon.'''
        n_sample = self.yearly_storm * self.t_years * self.nr_batch
        return storm.generate(
            fitted_storm=self.fitted_storm,
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
        )

        storm_count_end = storm_count + len(synth_start)

        synth_hs = hss[storm_count:storm_count_end]
        synth_direction = dirs[storm_count:storm_count_end]
        synth_duration = durs[storm_count:storm_count_end]
        synth_tp = tps[storm_count:storm_count_end]
        synth_end = synth_start + (synth_duration / 24)
        synth_gap = synth_start - np.roll(synth_end, 1)
        synth_gap[0] = 0.0

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
        )

        synth_recovery = shoreline.vector_calculate_recovery(
            gaps=synth_gap,
            rec_rate=self.rec_rate,
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
        '''Run the batched Monte-Carlo simulation stage (assumes init_slr/load_wave_data/detect_storms already ran).'''
        self.prepare_simulation()

        sim_count = 0
        error_count = 0
        while sim_count < self.nr_simulation:
            hss, durs, dirs, tps = self._generate_batch()
            storm_count = 0

            print(f'progress: {sim_count / self.nr_simulation * 100:.2f} %')
            for _ in range(self.nr_batch):
                self.track_time[sim_count], self.track_shoreline[sim_count], storm_count = self._simulate_one(hss, durs, dirs, tps, storm_count)

                row = self.compute_statistics(sim_count)

                try:
                    self.shoreline_stats[:, sim_count] = row.flatten()
                except ValueError:
                    sim_count -= 1
                    error_count += 1

                sim_count += 1
                if sim_count >= self.nr_simulation:
                    break

        return self.shoreline_stats

    def run(self):
        '''Run the full pipeline: SLR curve, wave data, storm detection, then batched simulation.'''
        t0 = time.time()

        print('Calculating Sea Level Rise ...')
        self.init_slr()

        print('Retrieve wave data ...')
        self.load_wave_data()

        print('Detecting storms ...')
        self.detect_storms()

        print('Starting the simulation ...')
        self.run_simulation()

        print(f'running for {time.time() - t0:2f} s')
        # return self.shoreline_stats, error_count
