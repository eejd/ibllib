from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from oneibl.one import ONE
from brainbox.processing import bincount2D
import alf.io as ioalf
import ibllib.plots as iblplt
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy.ma as ma
plt.ion()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# get the data from flatiron and the current folder
#one = ONE()
#eid = one.search(subject='ZM_1736', date='2019-08-09', number=4)
#D = one.load(eid[0], clobber=False, download_only=True)
#session_path = Path(D.local_path[0]).parent
session_path = Path(
    '/home/mic/Downloads/FlatIron/mainenlab/ZM_1735_2019-08-01_001/mnt/s0/Data/Subjects/ZM_1735/2019-08-01/001/alf')

# load objects
spikes = ioalf.load_object(session_path, 'spikes')
trials = ioalf.load_object(session_path, '_ibl_trials')
wheel = ioalf.load_object(session_path, '_ibl_wheel')


def bin_types(spikes, trials, wheel):

    T_BIN = 0.2  # [sec]

    # TO GET MEAN: bincount2D(..., weight=positions) / bincount2D(..., weight=None)

    reward_times = trials['feedback_times'][trials['feedbackType'] == 1]
    trial_start_times = trials['intervals'][:, 0]
    # trial_end_times = trials['intervals'][:, 1] #not working as there are
    # nans
    # compute raster map as a function of cluster number
    R1, times1, _ = bincount2D(
        spikes['times'], spikes['clusters'], T_BIN, weights=spikes['amps'])
    R2, times2, _ = bincount2D(
        reward_times, np.array(
            [0] * len(reward_times)), T_BIN)
    R3, times3, _ = bincount2D(
        trial_start_times, np.array(
            [0] * len(trial_start_times)), T_BIN)
    R4, times4, _ = bincount2D(wheel['times'], np.array(
        [0] * len(wheel['times'])), T_BIN, weights=wheel['position'])
    R5, times5, _ = bincount2D(wheel['times'], np.array(
        [0] * len(wheel['times'])), T_BIN, weights=wheel['velocity'])
    #R6, times6, _ = bincount2D(trial_end_times, np.array([0]*len(trial_end_times)), T_BIN)
    start = max([x for x in [times1[0], times2[0], times3[0], times4[0], times5[0]]])
    stop = min([x for x in [times1[-1], times2[-1],
                            times3[-1], times4[-1], times5[-1]]])
    time_points = np.linspace(start, stop, int((stop - start) / T_BIN))
    binned_data = {}
    binned_data['wheel_position'] = np.interp(
        time_points, wheel['times'], wheel['position'])
    binned_data['wheel_velocity'] = np.interp(
        time_points, wheel['times'], wheel['velocity'])
    binned_data['summed_spike_amps'] = R1[:, find_nearest(
        times1, start):find_nearest(times1, stop)]
    binned_data['reward_event'] = R2[0, find_nearest(
        times2, start):find_nearest(times2, stop)]
    binned_data['trial_start_event'] = R3[0, find_nearest(
        times3, start):find_nearest(times3, stop)]
    # binned_data['trial_end_event']=R6[0,find_nearest(times6,start):find_nearest(times6,stop)]
    # np.vstack([R1,R2,R3,R4])
    return binned_data

def color_attractor(binned_data, bounds):

    # obs_limit=1000 #else it's too slow  ​
    low, high = bounds

    X = binned_data['summed_spike_amps'][:, low:high].T
    Y = manifold.Isomap(n_components=3).fit_transform(X)

    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=abs(
        binned_data['wheel_velocity'][low:high]), cmap='binary')
    fig.colorbar(p)
    plt.title("Guido's motor cortex --> thalamus recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()


# for ZM_1735 use left probe only

