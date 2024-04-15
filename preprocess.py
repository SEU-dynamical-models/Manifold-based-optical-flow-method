import mne
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import yaml
import pandas as pd
from scipy.stats import zscore
# import hdf5storage as hdf
from mne_bids import BIDSPath, write_raw_bids


vhdr_data = r"data\sub-p08\ses-nyuecog01\ieeg\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_ieeg.vhdr"
json_data = r"data\sub-p08\ses-nyuecog01\ieeg\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_ieeg.json"
events_tsv_data = r"data\sub-p08\ses-nyuecog01\ieeg\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_events.tsv"
channels_tsv_data = r"data\sub-p08\ses-nyuecog01\ieeg\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_channels_renamed.tsv"
high_freq = 100
low_freq = 0.1
baseline_shift_start = -0.1
baseline_shift_end = 0

raw = mne.io.read_raw_brainvision(vhdr_data, preload=True)
# print(raw.info)

#from .json reading powerline frequency and sampling frequceny 从json文件读取工频和采样率
with open(json_data,'r',encoding='UTF-8') as f:
    json_info = json.load(f)
PLF = json_info["PowerLineFrequency"]
SF = round(json_info["SamplingFrequency"])

print(raw.info)
print(raw.info['ch_names'])

# raw.plot(scalings=4e-4,n_channels=30)
# plt.tight_layout()
# plt.show(block=True)#show raw data 展示原数据
# spectrum = raw.compute_psd()
# spectrum.plot()
# plt.tight_layout()
# plt.show(block=True)


hl_data = raw.filter(h_freq=high_freq, l_freq=low_freq)  # high-low pass filter 高-低通滤波
# hl_data.plot(scalings=4e-4,n_channels=60)
# plt.tight_layout()
# plt.show(block=True)  # show data after high-low pass filter 展示高-低通滤波后的数据
# spectrum_hl = hl_data.compute_psd()
# spectrum_hl.plot()
# plt.tight_layout()
# plt.show(block=True)


notch_data = hl_data.notch_filter(freqs=PLF)  # removing powerline noise 去除工频噪音
# notch_data.plot(scalings=4e-4,n_channels=30)
# plt.tight_layout()
# plt.show(block=True)  # show data after removing powerline noise 展示去除工频后的数据
# spectrum_pl = notch_data.compute_psd()
# spectrum_pl.plot()
# plt.tight_layout()
# plt.show(block=True)



channels_tsv = pd.read_csv(channels_tsv_data, sep='\t')
conditions = (channels_tsv['type'] == 'ECOG') & (channels_tsv['status'] == 'good')
selected_names = channels_tsv.loc[conditions, 'name'].tolist()
print("Selected electrode names:")
print(selected_names)
picks = [notch_data.ch_names.index(name) for name in selected_names if name in notch_data.ch_names]
drop_ch_data = notch_data.pick_channels([notch_data.ch_names[pick] for pick in picks])
print(drop_ch_data.info)
# drop_ch_data.plot(scalings=4e-4,n_channels=30)
# plt.tight_layout()
# plt.show(block=True)  # show data after removing powerline noise 展示去除工频后的数据
# spectrum_pl = drop_ch_data.compute_psd()
# spectrum_pl.plot()
# plt.tight_layout()
# plt.show(block=True)


# rereferencing with common average reference  通过全局平均参考进行重参考
rereferenced_data, ref_data = mne.set_eeg_reference(drop_ch_data, copy=True)
# rereferenced_data.plot(scalings=4e-4,n_channels=60)
# plt.tight_layout()
# plt.show(block=True)  # show the data after rereferencing 展示重参考后的数据
# spectrum_ref = rereferenced_data.compute_psd()
# spectrum_ref.plot()
# plt.tight_layout()
# plt.show(block=True)



# 数据分段
# print(rereferenced_data.annotations)
# print(rereferenced_data.annotations.duration)
# print(rereferenced_data.annotations.description)
# print(rereferenced_data.annotations.onset)
events = []
event_id = {}
events_info = pd.read_csv(events_tsv_data, delimiter='\t')
for row in events_info.index:
    onsetsamp = events_info.loc[row]["event_sample"]
    events.append([onsetsamp, 0 , events_info.loc[row]["trial_type"]])
    event_id.update({events_info.loc[row]["trial_name"]: events_info.loc[row]["trial_type"]})
    
print(np.array(events).shape, event_id)

epochs = mne.Epochs(rereferenced_data, events, event_id=event_id, tmin=-0.1, tmax=0.5, baseline=(baseline_shift_start, baseline_shift_end), preload=False)
print(epochs)
epochs.save('results\sub-p08\preprocessed_data\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01-epo.fif', overwrite=True)
epochs.plot(n_epochs=4)
plt.tight_layout()
plt.show(block=True)


# print(epochs[0])
# epochs[0].plot()
# plt.tight_layout()
# plt.show(block=True)



# evoked = epochs.average(by_event_type=True)
# for e in evoked:
#     print(e.comment)
#     e.save(f'sub-01/preprocessed_data/sub_01-{e.comment}-ave.fif', overwrite=True)
    # e.plot() # exclude=['PT01', 'PT02']
    # plt.show(block=True)



# evoked.plot_topo()
# plt.show(block=True)
# mne.viz.plot_compare_evokeds(evokeds=evoked, combine='mean', time_unit='ms')
# plt.show(block=True)


# mne.viz.plot_compare_evokeds(evokeds=evoked, picks=['C41', 'C42', 'C49', 'C50', 'C51', 'C52', 'C57', 'C58', 'C59', 'C60', 'C61', 'TP6', 'TP7', 'TP8', 'TP14', 'TP15', 'TP16', 'TP22', 'TP23', 'TP24', 'TP30', 'TP31', 'P03', 'P10', 'P11', 'P12', 'P13', 'C39', 'C40', 'TP32', 'C37', 'C38', 'P05', 'P06', 'P07', 'P14', 'P15', 'P04'], combine='mean')
# # picks=['C41', 'C42', 'C49', 'C50', 'C51', 'C52', 'C57', 'C58', 'C59', 'C60', 'C61', 'TP6', 'TP7', 'TP8', 'TP14', 'TP15', 'TP16', 'TP22', 'TP23', 'TP24', 'TP30', 'TP31', 'P03', 'P10', 'P11', 'P12', 'P13', 'C39', 'C40', 'TP32', 'C37', 'C38', 'P05', 'P06', 'P07', 'P14', 'P15', 'P04']  picks=['F61', 'F62', 'F52', 'F53', 'F59', 'F60', 'F49', 'F50', 'F51', 'F57', 'F58', 'PT25', 'PT26', 'PT27', 'PT28', 'PT33', 'PT34', 'PT35', 'PT41', 'PT42', 'PT43', 'PT44', 'PT36', 'PT38', 'PT39', 'PT45', 'PT46', 'PT47', 'PT48', 'IH6', 'IH7', 'IH8']
# plt.show(block=True)