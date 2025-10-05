# -*- coding: utf-8 -*-
# 读取ECoG电极坐标, 并重建流形
# Author: Xi Wang
# Date: 28 February 2024
# Email: 2308180834@qq.com

"""
这段代码主要用于从TSV文件中提取坐标数据,并绘制三维图形。通过读取一个YAML配置文件来获取参数,然后在指定的文件夹结构中查找并读取TSV文件。然后,它根据提供的ECOG名称列表筛选出相应的坐标数据,并使用matplotlib绘制三维散点图。
"""

import os
import mne
import pyvista as pv
from matplotlib.tri import Triangulation
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

# 数据路径和结果保存路径
data_path = '/fred/oz284/mc/data/ds004080'
results_path = '/fred/oz284/mc/results/ds004080'


if not os.path.exists(results_path):
    os.mkdir(results_path)

# 获取所有的被试文件夹
subfolders = [name for name in os.listdir(data_path) if name.startswith('sub')]
subfolders = sorted(subfolders)
print(len(subfolders))


for subfolder in subfolders:
    results_subfolder_path = f"{results_path}/{subfolder}"
    data_subfolder_path = f"{data_path}/{subfolder}"

    filelist = os.listdir(data_subfolder_path)
    ses = filelist[0]
    data_subfolder_path = f"{data_path}/{subfolder}/{ses}/ieeg"

    save_path = f"{results_subfolder_path}/{subfolder}_reconstructed_surface.ply"

    electrodes_path = f"{data_subfolder_path}/{subfolder}_{ses}_electrodes.tsv"
    # channels_path = f"{data_subfolder_path}/{subfolder}_ses-1_electrodes.tsv"

    # 读取电极TSV文件
    electrodes_tsv = pd.read_csv(electrodes_path, sep='\t')

    # 只选择group为'grid'的电极
    conditions = (electrodes_tsv['group'] == 'grid')
    # & (channels_tsv['status'] == 'good') & (channels_tsv['status_description'] == 'included')  & (channels_tsv['group'] == 'grid')
    selected_names = electrodes_tsv.loc[conditions, 'name'].tolist()

    coordinates = []
    points = []
    # 逐行读取TSV文件,提取坐标
    with open(electrodes_path, 'r') as file:
        lines = file.readlines()[1:]  # 跳过表头
        for line in lines:
            name, x, y, z = line.strip().split('\t')[:4]
            if x != 'n/a':
                if name in selected_names:
                    points.append((float(x), float(y), float(z)))
                    coordinates.append((name, float(x), float(y), float(z)))

    # 绘制三维散点图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for name, x, y, z in coordinates:
    #     ax.scatter(x, y, z, color='blue')
    #     ax.text(x, y, z, name, color='black', fontsize=5)
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # plt.show()


    electrodes = points  

    # 创建点云对象
    cloud = pv.PolyData(points)
    # 基于点云进行二维Delaunay三角剖分,生成表面网格
    mesh = cloud.delaunay_2d()
    # mesh = cloud.delaunay_3d()
    # mesh = cloud.reconstruct_surface()

    # 对网格进行平滑和细分处理
    mesh1 = mesh.smooth(n_iter=100)           # 平滑操作
    # mesh1 = mesh1.subdivide(3)
    mesh1 = mesh1.subdivide(3, 'butterfly')    # 细分网格
    mesh1 = mesh1.smooth(n_iter=100)           # 再次平滑
    # 保存重建后的表面为PLY文件
    mesh1.save(save_path)
    mesh1.save(f"{results_path}/{subfolder}_reconstructed_surface.ply")
    print(f"重建后的表面有顶点：{len(mesh1.points)}, 三角形有{len(mesh1.faces.reshape(-1, 4)[:, 1:])}")