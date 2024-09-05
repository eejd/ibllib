import logging

import numpy as np

from one.alf.spec import is_session_path
import one.alf.io as alfio
from one.api import ONE


_logger = logging.getLogger('ibllib')

REQUIRED_FIELDS = ['choice', 'contrastLeft', 'contrastRight', 'correct',
                   'errorCueTrigger_times', 'errorCue_times', 'feedbackType', 'feedback_times',
                   'firstMovement_times', 'goCueTrigger_times', 'goCue_times', 'intervals',
                   'itiIn_times', 'phase', 'position', 'probabilityLeft', 'quiescence',
                   'response_times', 'rewardVolume', 'stimFreezeTrigger_times',
                   'stimFreeze_times', 'stimOffTrigger_times', 'stimOff_times',
                   'stimOnTrigger_times', 'stimOn_times', 'valveOpen_times',
                   'wheel_moves_intervals', 'wheel_moves_peak_amplitude',
                   'wheel_position', 'wheel_timestamps']


class TaskQCExtractor:
    def __init__(self, session_path, one=None, download_data=False, bpod_only=False,
                 sync_collection=None, sync_type=None, task_collection=None):
        """
        A class for extracting the task data required to perform task quality control.
        :param session_path: a valid session path
        :param one: an instance of ONE, used to download the raw data if download_data is True
        :param download_data: if True, any missing raw data is downloaded via ONE
        :param bpod_only: extract from raw Bpod data only, even for FPGA sessions
        """
        if not is_session_path(session_path):
            raise ValueError('Invalid session path')
        self.session_path = session_path
        self.one = one
        self.log = _logger

        self.data = None
        self.settings = None
        self.raw_data = None
        self.frame_ttls = self.audio_ttls = self.bpod_ttls = None
        self.type = None
        self.wheel_encoding = None
        self.bpod_only = bpod_only
        self.sync_collection = sync_collection or 'raw_ephys_data'
        self.sync_type = sync_type
        self.task_collection = task_collection or 'raw_behavior_data'

        if download_data:
            self.one = one or ONE()

    @staticmethod
    def rename_data(data):
        """Rename the extracted data dict for use with TaskQC
        Splits 'feedback_times' to 'errorCue_times' and 'valveOpen_times'.
        NB: The data is not copied before making changes
        :param data: A dict of task data returned by the task extractors
        :return: the same dict after modifying the keys
        """
        # Expand trials dataframe into key value pairs
        trials_table = data.pop('table', None)
        if trials_table is not None:
            data = {**data, **alfio.AlfBunch.from_df(trials_table)}
        correct = data['feedbackType'] > 0
        # get valve_time and errorCue_times from feedback_times
        if 'errorCue_times' not in data:
            data['errorCue_times'] = data['feedback_times'].copy()
            data['errorCue_times'][correct] = np.nan
        if 'valveOpen_times' not in data:
            data['valveOpen_times'] = data['feedback_times'].copy()
            data['valveOpen_times'][~correct] = np.nan
        if 'wheel_moves_intervals' not in data and 'wheelMoves_intervals' in data:
            data['wheel_moves_intervals'] = data.pop('wheelMoves_intervals')
        if 'wheel_moves_peak_amplitude' not in data and 'wheelMoves_peakAmplitude' in data:
            data['wheel_moves_peak_amplitude'] = data.pop('wheelMoves_peakAmplitude')
        data['correct'] = correct
        diff_fields = list(set(REQUIRED_FIELDS).difference(set(data.keys())))
        for miss_field in diff_fields:
            data[miss_field] = data['feedback_times'] * np.nan
        if len(diff_fields):
            _logger.warning(f'QC extractor, missing fields filled with NaNs: {diff_fields}')
        return data
