import mne
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import yaml
import pandas as pd
import pyvista as pv
from scipy.stats import zscore
# import hdf5storage as hdf
from mne_bids import BIDSPath, write_raw_bids
from scipy.interpolate import Rbf




vhdr_data = r"data\sub-p08\ses-nyuecog01\ieeg\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_ieeg.vhdr"
json_data = r"data\sub-p08\ses-nyuecog01\ieeg\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_ieeg.json"
events_tsv_data = r"data\sub-p08\ses-nyuecog01\ieeg\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_events.tsv"
channels_tsv_data = r"data\sub-p08\ses-nyuecog01\ieeg\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_channels.tsv"
electrodes_tsv_path = r"data\sub-p08\ses-nyuecog01\ieeg\sub-p08_ses-nyuecog01_acq-clinical_electrodes.tsv"

epochs_path = r"results\sub-p08\preprocessed_data\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01-epo.fif"
reconstruct_surface_save_path = r"results\sub-p08\preprocessed_data\sub-p08_reconstructed_surface.ply"


trials = ["VERTICAL-L-R-1", "VERTICAL-L-R-2", "VERTICAL-L-R-3", "VERTICAL-L-R-4", "VERTICAL-L-R-5", "VERTICAL-L-R-6", "VERTICAL-L-R-7", "VERTICAL-L-R-8", "VERTICAL-L-R-9", "VERTICAL-L-R-10", "VERTICAL-L-R-11", "VERTICAL-L-R-12", "VERTICAL-L-R-13", "VERTICAL-L-R-14", "VERTICAL-L-R-15", "VERTICAL-L-R-16", "VERTICAL-L-R-17", "VERTICAL-L-R-18", "VERTICAL-L-R-19", "VERTICAL-L-R-20", "VERTICAL-L-R-21", "VERTICAL-L-R-22", "VERTICAL-L-R-23", "VERTICAL-L-R-24", "VERTICAL-L-R-25", "VERTICAL-L-R-26", "VERTICAL-L-R-27", "VERTICAL-L-R-28", "DIAGONAL-RD-LU-1", "DIAGONAL-RD-LU-2", "DIAGONAL-RD-LU-3", "DIAGONAL-RD-LU-4", "DIAGONAL-RD-LU-5", "DIAGONAL-RD-LU-6", "DIAGONAL-RD-LU-7", "DIAGONAL-RD-LU-8", "DIAGONAL-RD-LU-9", "DIAGONAL-RD-LU-10", "DIAGONAL-RD-LU-11", "DIAGONAL-RD-LU-12", "BLANK", "HORIZONTAL-U-D-1", "HORIZONTAL-U-D-2", "HORIZONTAL-U-D-3", "HORIZONTAL-U-D-4", "HORIZONTAL-U-D-5", "HORIZONTAL-U-D-6", "HORIZONTAL-U-D-7", "HORIZONTAL-U-D-8", "HORIZONTAL-U-D-9", "HORIZONTAL-U-D-10", "HORIZONTAL-U-D-11", "HORIZONTAL-U-D-12", "HORIZONTAL-U-D-13", "HORIZONTAL-U-D-14", "HORIZONTAL-U-D-15", "HORIZONTAL-U-D-16", "HORIZONTAL-U-D-17", "HORIZONTAL-U-D-18", "HORIZONTAL-U-D-19", "HORIZONTAL-U-D-20", "HORIZONTAL-U-D-21", "HORIZONTAL-U-D-22", "HORIZONTAL-U-D-23", "HORIZONTAL-U-D-24", "HORIZONTAL-U-D-25", "HORIZONTAL-U-D-26", "HORIZONTAL-U-D-27", "HORIZONTAL-U-D-28", "DIAGONAL-LD-RU-1", "DIAGONAL-LD-RU-2", "DIAGONAL-LD-RU-3", "DIAGONAL-LD-RU-4", "DIAGONAL-LD-RU-5", "DIAGONAL-LD-RU-6", "DIAGONAL-LD-RU-7", "DIAGONAL-LD-RU-8", "DIAGONAL-LD-RU-9", "DIAGONAL-LD-RU-10", "DIAGONAL-LD-RU-11", "DIAGONAL-LD-RU-12", "DIAGONAL-LU-RD-1", "DIAGONAL-LU-RD-2", "DIAGONAL-LU-RD-3", "DIAGONAL-LU-RD-4", "DIAGONAL-LU-RD-5", "DIAGONAL-LU-RD-6", "DIAGONAL-LU-RD-7", "DIAGONAL-LU-RD-8", "DIAGONAL-LU-RD-9", "DIAGONAL-LU-RD-10", "DIAGONAL-LU-RD-11", "DIAGONAL-LU-RD-12", "DIAGONAL-RU-LD-1", "DIAGONAL-RU-LD-2", "DIAGONAL-RU-LD-3", "DIAGONAL-RU-LD-4", "DIAGONAL-RU-LD-5", "DIAGONAL-RU-LD-6", "DIAGONAL-RU-LD-7", "DIAGONAL-RU-LD-8", "DIAGONAL-RU-LD-9", "DIAGONAL-RU-LD-10", "DIAGONAL-RU-LD-11", "DIAGONAL-RU-LD-12"]

epochs = mne.read_epochs(epochs_path, preload=False)
mesh = pv.read(reconstruct_surface_save_path)

channels_tsv = pd.read_csv(channels_tsv_data, sep='\t')
conditions = (channels_tsv['type'] == 'ECOG') & (channels_tsv['status'] == 'good')
# & (channels_tsv['status'] == 'good') & (channels_tsv['status_description'] == 'included')  & (channels_tsv['group'] == 'grid')
selected_names = channels_tsv.loc[conditions, 'name'].tolist()


coordinates = []
points = []
with open(electrodes_tsv_path, 'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        name, x, y, z = line.strip().split('\t')[:4]
        # if x != 'n/a':
        if name in selected_names:
            points.append((float(x), float(y), float(z)))
            coordinates.append((name, float(x), float(y), float(z)))
vertices_array = mesh.points
points = np.array(points)
print(points.shape)


for trial in trials:
    epochs_trial = epochs[trial]
    for i, potentials in enumerate(epochs_trial):
        potentials_save_path = "results/sub-p08/preprocessed_data/trials_potentials/sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_" + trial + "_0{}_potentials.csv".format(i+1)
        # print(potentials.shape)

        potentials = potentials.T  # (t, ecog_nums)
        # print(potentials.shape)

        interpolated_values = []

        for potentials_frame in potentials:
            # 创建径向基函数插值对象
            rbf                       = Rbf(points[:, 0], points[:, 1], points[:, 2], potentials_frame)
            interpolated_values_frame = rbf(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2])
            # print(interpolated_values_frame.shape)
            interpolated_values.append(interpolated_values_frame)

        interpolated_values = np.array(interpolated_values)   # (t, ecog_nums)
        print(interpolated_values.shape)

        pd.DataFrame(interpolated_values).to_csv(potentials_save_path)    # (t, ecog_nums)





# print(epochs[0])
# potentials = epochs[0].get_data()
# print(potentials.shape)
# # epochs.plot(n_epochs=4)
# # plt.tight_layout()
# # plt.show(block=True)
# # print(all_epochs)
# # all_epochs.plot()


# channels_tsv = pd.read_csv(channels_tsv_data, sep='\t')
# conditions = (channels_tsv['type'] == 'ECOG') & (channels_tsv['status'] == 'good')
# # & (channels_tsv['status'] == 'good') & (channels_tsv['status_description'] == 'included')  & (channels_tsv['group'] == 'grid')
# selected_names = channels_tsv.loc[conditions, 'name'].tolist()


# coordinates = []
# points = []
# with open(electrodes_tsv_path, 'r') as file:
#     lines = file.readlines()[1:]
#     for line in lines:
#         name, x, y, z = line.strip().split('\t')[:4]
#         # if x != 'n/a':
#         if name in selected_names:
#             points.append((float(x), float(y), float(z)))
#             coordinates.append((name, float(x), float(y), float(z)))

# # Create point cloud object 创建点云对象
# cloud = pv.PolyData(points)

# # Perform 2D Delaunay triangulation 执行三角网格重建
# mesh = cloud.delaunay_2d()
# # points = pv.wrap(points)
# # mesh = points.reconstruct_surface()

# # Smooth and subdivide the mesh 平滑操作和细分网格
# mesh = mesh.smooth(n_iter=100)           # 平滑操作
# mesh = mesh.subdivide(3, 'butterfly')    # 细分网格
# mesh = mesh.smooth(n_iter=100)           # 平滑操作

# # Get all points on the surface 获取曲面上的所有点
# # points     = mesh.points
# # num_points = points.shape[0]
# # print(points.shape)

# mesh.save(reconstruct_surface_save_path)


# p = pv.Plotter()
# p.add_mesh(mesh, show_edges=True)
# p.add_mesh(cloud.points, color="green", show_edges=True)
# p.show()


# potentials = potentials.reshape((potentials.shape[1], potentials.shape[2])) # (ecog_nums, t)
# potentials = potentials.T  # (t, ecog_nums)
# print(potentials.shape)


# # surface        = pv.read(surface_path)
# vertices_array = mesh.points
# points = np.array(points)
# interpolated_values = []

# for potentials_frame in potentials:
#     # 创建径向基函数插值对象
#     rbf                       = Rbf(points[:, 0], points[:, 1], points[:, 2], potentials_frame)
#     interpolated_values_frame = rbf(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2])

#     # print(interpolated_values_frame.shape)
#     interpolated_values.append(interpolated_values_frame)

# interpolated_values = np.array(interpolated_values)   # (t, ecog_nums)
# print(interpolated_values.shape)

# pd.DataFrame(interpolated_values).to_csv(potentials_save_path)    # (t, ecog_nums)