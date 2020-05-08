import ibllib.qc.bpodqc_metrics as bpodqc
import ibllib.qc.oneqc_metrics as oneqc
from ibllib.qc.bpodqc_extractors import load_bpod_data
from ibllib.qc.oneutils import random_ephys_session
from oneibl.one import ONE

one = ONE()

eid, det = random_ephys_session()


def build_extended_qc_frame(eid, data=None):
    # Get bpod and one qc frames
    data = load_bpod_data(eid)
    extended_qc = {}
    one_frame = oneqc.get_oneqc_metrics_frame(eid, data=data, apply_criteria=True)
    bpod_frame = bpodqc.get_bpodqc_metrics_frame(eid, data=data, apply_criteria=True)
    # Make average bool pass

    # aggregate them
    extended_qc.update(one_frame)
    extended_qc.update(bpod_frame)
    return extended_qc

# ad-hoc aggregation for some variables
extended_qc = {
    'feedback': np.sum(qc_frame.n_feedback != 0) / ntrials,
}
# here we average all bools
for k in qc_frame.keys():
    if qc_frame[k].dtype == np.dtype('bool'):
        extended_qc[k] = np.mean(qc_frame[k])
    elif k.endswith('_times'):
        # Todo remove intervals_start from there if it's an absolute time
        # std + mean ?
        pass
