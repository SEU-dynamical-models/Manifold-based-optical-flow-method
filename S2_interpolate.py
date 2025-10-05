# -*- coding: utf-8 -*-
# 读取ECoG电极坐标, 并重建流形
# Author: Xi Wang
# Date: 28 February 2024
# Email: 2308180834@qq.com

import mne
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import os
import re
import sys
from scipy.signal import hilbert

# 设置插值的起止时间（单位：秒）
start_time = 2.009  # 开始时间
end_time = 2.2      # 结束时间

def interpolation(surface_path, ieeg_data_array, coordinates, start_sample, end_sample, save_path, ifsave):
    '''
    对电极电势数据进行空间插值
    参数:
        surface_path: 重建表面文件路径
        ieeg_data_array: 电极电势数据, shape=(时间, 电极数)
        coordinates: 电极坐标 (x, y, z), shape=(电极数, 3)
        start_sample, end_sample: 采样点范围
        save_path: 插值结果保存路径
        ifsave: 是否保存结果
    '''
    ieeg_data_array = ieeg_data_array[start_sample:end_sample]

    # 读取重建表面，获取插值点坐标
    surface = pv.read(surface_path)
    vertices_array = surface.points
    
    interpolated_values = []

    # 对每个时间点的数据进行插值
    for ieeg_data_frame in ieeg_data_array:
        # 创建径向基函数插值对象
        rbf = Rbf(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], ieeg_data_frame)
        interpolated_values_frame = rbf(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2])
        interpolated_values.append(interpolated_values_frame)

    interpolated_values = np.array(interpolated_values)   # (t, 点数)
    print(f"插值后的形状(t, ecog): {interpolated_values.shape}")

    # 保存插值结果
    if ifsave is True:
        pd.DataFrame(interpolated_values).to_csv(save_path)

def get_subfolders(directory):
    '''获取指定目录下的所有子文件夹名称'''
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

if __name__ == "__main__":
    ifsave = True
    data_path = '/fred/oz284/mc/data/ds004080'
    results_path = '/fred/oz284/mc/results/ds004080'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    subfolder = sys.argv[1]

    surface_path = f"{results_path}/{subfolder}/{subfolder}_reconstructed_surface.ply"
    results_subfolder_path = f"{results_path}/{subfolder}"
    data_subfolder_path = f"{data_path}/{subfolder}"

    filelist = os.listdir(data_subfolder_path)
    ses = filelist[0]
    data_subfolder_path = f"{data_path}/{subfolder}/{ses}/ieeg"

    if not os.path.exists(results_subfolder_path):
        os.mkdir(results_subfolder_path)

    # 正则表达式匹配 run 编号
    run_pattern = re.compile(r'run-(\d{6})')
    run_numbers = set()
    for root, dirs, files in os.walk(data_subfolder_path):
        for file in files:
            match = run_pattern.search(file)
            if match:
                run_numbers.add(match.group(1))

    # 输出所有找到的 run 编号
    print("Found run numbers:", sorted(run_numbers))
    for run_num in run_numbers:
        run_path = f"{results_subfolder_path}/run-{run_num}"
        if not os.path.exists(run_path):
            os.mkdir(run_path)
        
        trials_folder_names = get_subfolders(run_path)
        print(trials_folder_names)
        # pattern2 用于提取刺激电极名称
        pattern2 = re.compile(r'(.*?)-(.*)')
        for trial_name in trials_folder_names:
            match2 = pattern2.search(trial_name)
            ecog1 = match2.group(1)
            ecog2 = match2.group(2)
            print(ecog1, ecog2)
            evoked_path = f"{run_path}/{trial_name}/{subfolder}_{ses}_task-SPESclin_run-{run_num}-{trial_name}-ave.fif"

            # 读取诱发数据
            evoked = mne.read_evokeds(evoked_path, 0)
            print(evoked.ch_names)
            print("evoked.data nan: ", np.sum(np.isnan(evoked.data)))
            start_sample = int(start_time * evoked.info['sfreq'])  # 转换为采样点
            end_sample = int(end_time * evoked.info['sfreq'])      # 转换为采样点

            electrodes_path = f"{data_subfolder_path}/{subfolder}_{ses}_electrodes.tsv"
            channels_path = f"{data_subfolder_path}/{subfolder}_{ses}_task-SPESclin_run-{run_num}_channels.tsv"

            # 读取通道信息，筛选有效ECOG电极
            channels_tsv = pd.read_csv(channels_path, sep='\t')
            conditions = (
                (channels_tsv['type'] == 'ECOG') &
                (channels_tsv['status'] == 'good') &
                (channels_tsv['status_description'] == 'included') &
                (channels_tsv['group'] == 'grid')
            )
            selected_names = channels_tsv.loc[conditions, 'name'].tolist()
            print(selected_names)
            # 移除刺激电极
            if ecog1 in selected_names:
                selected_names.remove(ecog1)
                print(f'{ecog1} removed from the list.')
            else:
                print(f'{ecog1} not found in the list.')
            if ecog2 in selected_names:
                selected_names.remove(ecog2)
                print(f'{ecog2} removed from the list.')
            else:
                print(f'{ecog2} not found in the list.')

            # 读取电极坐标
            electrodes_tsv = pd.read_csv(electrodes_path, sep='\t')
            filtered_data = electrodes_tsv[electrodes_tsv['name'].isin(selected_names)]
            coordinates = filtered_data[['x', 'y', 'z']].values
            print(f"coordinates shape: {coordinates.shape}")

            # 获取每个电极的电势数据
            ieeg_data = []
            for name in selected_names:
                ieeg_data.append(evoked.get_data(name))
            print(f"ieeg_data shape:{np.array(ieeg_data).shape}")
            print("ieeg_data nan: ", np.sum(np.isnan(ieeg_data)))

            ieeg_data = np.array(ieeg_data)
            ieeg_data = ieeg_data.T  # 转置为 (时间, 电极数)

            save_folder = f"{run_path}/{trial_name}"
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            save_path = f"{save_folder}/{subfolder}_{ses}_task-SPESclin_run-{run_num}-{trial_name}-ave-interpolation_data.csv"

            # 执行插值
            interpolation(surface_path, ieeg_data, coordinates, start_sample, end_sample, save_path, ifsave)