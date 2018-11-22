import matplotlib.pyplot as plt
import pandas as pd

from oneibl.one import ONE
from ibllib.time import isostr2date

one = ONE()
ac = one._alyxClient

# this gets all weighings for all subjects
wei = ac.rest('weighings', 'list')

# this gets the weighings for one subject
subject = '437'
wei = pd.DataFrame(ac.rest('weighings', 'list', '?nickname=' + subject))
plt.plot_date(wei['date_time'].apply(isostr2date), wei['weight'])

# to list administrations for one subject, it is better to use the subjects endpoint
sub_info = ac.rest('subjects', 'read', '437')
wei = pd.DataFrame(sub_info['weighings'])
wei['date_time'].apply(isostr2date)
wei.sort_values('date_time', inplace=True)
plt.plot(wei.date_time, wei.weight)
