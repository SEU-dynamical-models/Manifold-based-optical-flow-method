import pickle
import numpy as np
import pyvista as pv
import csv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ast
import pyvista as pv
import yaml
pv.global_theme.allow_empty_mesh = True
# V_k (3, 6202)
# e (3101, 6)


# 加载CSV文件数据
def load_data(file_path):
    return pd.read_csv(file_path, header='infer', index_col=0).values

# 将速度场数据 V_k 转换为速度向量
def process_V_k(V_k, e):
    """
    将速度场数据 V_k 转换为速度向量
    
    参数:
    - V_k (np.ndarray): 形状为 (k, 2*n) 的速度场数据, k 表示时间步数, n 表示点的个数
    - e (np.ndarray): 形状为 (n, 2, 3) 的基底向量
    
    返回:
    - V_k_coord (np.ndarray): 形状为 (k, n, 3) 的速度向量数据
    """
    point_num = len(e) # 曲面上点的个数
    V_k_array = []  # 将 V_k 重新组织成 (k, n, 2) 的形状

    for k in range(len(V_k)):
        V = []
        V_index = V_k[k]
        for i in range(point_num):
            V.append([V_index[i], V_index[i + point_num]])  # 构建每个点在基底方向的速度分量 [V_x, V_y]
        V_k_array.append(V)
        # V = np.array(V)
    V_k_array = np.array(V_k_array)
    # print(V_k_array.shape)

    V_k_coord = []
    for k in range(len(V_k_array)):
        V_index = V_k_array[k]
        V_arrow = []
        for i in range(point_num):
            # 计算每个点在基底方向上的速度矢量
            V_1 = V_index[i][0] * e[i][0]
            V_2 = V_index[i][1] * e[i][1]
            # if i == 0:
            #     print(V_index[i][0], e[i][0])
            #     print(V_1, V_2)
            V_arrow.append(V_1 + V_2)
        V_k_coord.append(V_arrow)

    return V_k_coord


# 在曲面上绘制速度场
def plot_velocity_fields_and_singularity_points(surface, singularity_points, potential, velocity, plot_type='Raw'):
    """
    在曲面上绘制速度场和临界点
    
    参数:
    - surface (pv.PolyData): 曲面数据
    - singularity_points (np.ndarray): 临界点的坐标
    - potential (np.ndarray): 曲面上每个点的电势值
    - velocity (np.ndarray): 曲面上每个点的速度向量
    - plot_type (str, 可选): 绘图类型, 可选 'Raw' 或 'Scaled'
    """
    # 设置曲面上每个点的电势值和速度向量
    surface['Potentials'] = potential
    surface['V']          = velocity

    cmin = np.min(potential)
    cmax = np.max(potential)

    if plot_type == 'Raw':
        # 根据速度向量的模长设置箭头的缩放比例scale
        lengths            = np.linalg.norm(velocity, axis=1)
        max_value          = np.max(lengths)
        scale_factor       = 10 / max_value
        scaled_lengths     = lengths * scale_factor
        surface['V_scale'] = scaled_lengths
        args               = {
            'tolerance': 0,
            # 'tolerance': 0.01,
            'factor'   : 0.4,
            'scale'    : 'V_scale',
            'orient'   : "V"
        }
    elif plot_type == 'Scaled':
        # scale设置为相同的数值,速度场仅显示方向,不体现变化大小
        surface['V_scale'] = np.ones((len(velocity), )) * 2
        args = {
            'tolerance': 0,
            # 'tolerance': 0.01,
            'factor': 0.7,
            'scale': 'V_scale',
            'orient': "V"
        }
    else:
        print("Wrong Type!")

    # 绘制箭头和临界点
    p      = pv.Plotter()
    arrows = surface.glyph(**args)
    p.add_mesh(arrows, color="black")
    p.add_mesh(pv.PolyData(singularity_points), color='red', point_size=5)
    # p.add_mesh(surface, scalars="Potentials", cmap="terrain", opacity=0.5, smooth_shading=True)
    # clim
    p.add_mesh(surface, scalars="Potentials", cmap="terrain", clim=[cmin, cmax], show_edges=False, smooth_shading=False)
    p.show()


# 在曲面上绘制速度场和临界点的动态 GIF 图像
def plot_velocity_fields_and_singularity_points_gif(surface, potentials, V_k_coord, singularity_points, velocity_fields_gif_path):
    """
    在曲面上绘制速度场和临界点的动态 GIF 图像
    
    参数:
    - surface (pv.PolyData): 曲面数据
    - potentials (list): k个时间步的电势值数组
    - V_k_coord (list): k个时间步的速度场坐标数组
    - singularity_points (list): k个时间步的临界点坐标数组
    """
    # 计算电势值的最大最小值
    cmin = np.min(potentials)
    cmax = np.max(potentials)

    # 计算速度场的缩放因子
    lengths            = [np.linalg.norm(v_k, axis=1) for v_k in V_k_coord]
    max_length         = np.max(lengths)
    scale_factor       = 20 / max_length
    surface['V_scale'] = lengths[0] * scale_factor

    # 创建 GIF 动画绘制器
    p = pv.Plotter()
    p.open_gif(velocity_fields_gif_path)

    # 添加第一帧的速度场和临界点
    surface['Potentials'] = potentials[0]
    surface['V']          = V_k_coord[0]
    args                  = {
        'tolerance': 0.01,
        'factor'   : 0.4,
        'scale'    : 'V_scale',
        'orient'   : 'V'
    }
    arrows = surface.glyph(**args)
    actor1 = p.add_mesh(arrows, color='black')
    actor2 = p.add_mesh(pv.PolyData(singularity_points[0]), color='red', point_size=10)
    p.add_mesh(surface, scalars="Potentials", cmap="terrain", clim=[cmin, cmax], show_edges=False, smooth_shading=False)

    # 逐帧绘制动画
    print('Orient the view, then press "q" to close window and produce movie')
    p.show(auto_close=False)
    for i in range(len(V_k_coord)):
        # if len(singularity_points[i]) == 0:
        #     continue
        p.remove_actor(actor1)
        p.remove_actor(actor2)

        surface['Potentials'] = potentials[i]
        surface['V']          = V_k_coord[i]
        surface['V_scale']    = lengths[i] * scale_factor
        arrows                = surface.glyph(**args)
        actor1                = p.add_mesh(arrows, color='black')
        actor2                = p.add_mesh(pv.PolyData(singularity_points[i]), color='red', point_size=10)
        p.add_mesh(surface, scalars="Potentials", cmap="terrain", clim=[cmin, cmax], show_edges=False, smooth_shading=False)
        p.write_frame()

    p.close()


def plot_velocity_fields_and_singularity_points_and_true_singularity_points_gif(surface, potentials, V_k_coord, singularity_points, true_singularity_points, velocity_fields_gif_path):
    """
    在曲面上绘制速度场、临界点和真实临界点的动态 GIF 图像
    
    参数:
    surface (pv.PolyData): 曲面数据
    potentials (list): k个时间步的电势值数组
    V_k_coord (list): k个时间步的速度场坐标数组
    singularity_points (list): k个时间步的临界点坐标数组
    true_singularity_points (list): k个时间步的真实临界点坐标数组
    """
    # 计算电势值的最大最小值,用来设置色条的范围
    cmin = np.min(potentials)
    cmax = np.max(potentials)

    # 计算速度场的缩放因子
    lengths            = [np.linalg.norm(v_k, axis=1) for v_k in V_k_coord]
    max_length         = np.max(lengths)
    scale_factor       = 20 / max_length
    surface['V_scale'] = lengths[0] * scale_factor
    
    # 创建 GIF 动画绘制器
    p = pv.Plotter()
    p.open_gif(velocity_fields_gif_path)

    # 添加第一帧的速度场、临界点和真实临界点
    surface['Potentials'] = potentials[0]
    surface['V']          = V_k_coord[0]
    args                  = {
        'tolerance': 0.01,
        'factor'   : 0.4,
        'scale'    : 'V_scale',
        'orient'   : 'V'
    }
    arrows = surface.glyph(**args)
    actor1 = p.add_mesh(arrows, color='black')
    actor2 = p.add_mesh(pv.PolyData(singularity_points[0]), color='red', point_size=10)
    actor3 = p.add_mesh(pv.PolyData(true_singularity_points[0]), color='black', point_size=10)
    p.add_mesh(surface, scalars="Potentials", cmap="terrain", clim=[cmin, cmax], show_edges=False, smooth_shading=False)

    # 逐帧绘制动画
    print('Orient the view, then press "q" to close window and produce movie')
    p.show(auto_close=False)
    for i in range(len(V_k_coord)):
        # if len(singularity_points[i]) == 0:
        #     continue
        p.remove_actor(actor1)
        p.remove_actor(actor2)
        p.remove_actor(actor3)

        surface['Potentials'] = potentials[i]
        surface['V']          = V_k_coord[i]
        surface['V_scale']    = lengths[i] * scale_factor
        arrows                = surface.glyph(**args)
        actor1                = p.add_mesh(arrows, color='black')
        actor2                = p.add_mesh(pv.PolyData(singularity_points[i]), color='red', point_size=10)
        actor3                = p.add_mesh(pv.PolyData(true_singularity_points[i]), color='black', point_size=10)
        p.add_mesh(surface, scalars="Potentials", cmap="terrain", clim=[cmin, cmax], show_edges=False, smooth_shading=False)

        p.write_frame()

    p.close()

if "__main__" == __name__:
    with open("./config/config.yaml", 'r', encoding='UTF-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # general_params = config['general']
    data_params    = config['sub_01']

    # lambda_       = general_params['lambda_']  # 光流法正则化参数(速度场平滑项参数)0.01
    # eps           = general_params['eps']    # 判断速度是否为0的误差
    # time_steps    = general_params['time_steps'] # 时间步数
    # processes_num = general_params['processes_num']  # 进程池大小

    surface_path             = data_params['surface_path']  # 三角剖分的表面文件路径
    potentials_path          = data_params['potentials_path']  # 电势数据文件路径
    e_path                   = data_params['e_path']  # 基底向量保存路径
    V_k_path                 = data_params['V_k_path']  # 速度场保存路径
    singularity_points_path  = data_params['singularity_points_path']  # 检测到的临界点保存路径
    velocity_fields_gif_path = data_params['velocity_fields_gif_path']  # 光流场绘制保存路径

    # e_path          = './optical_flow/e.csv'
    # V_k_path        = './optical_flow/V_k.csv'
    # surface_path    = './reconstructed_surface/sub-p01_reconstructed_surface.ply'
    # potentials_path = './interpolation_data/sub-p01_ses-umcuiemu01_task-prf_acq-clinical_run-01_ieeg.csv'

    # e_path          = './simulated_data/e.csv'
    # V_k_path        = './simulated_data/V_k.csv'
    # surface_path    = './simulated_data/simulated_surface.ply'
    # potentials_path = './simulated_data/simulated_potentials.csv'

    surface    = pv.read(surface_path)
    e          = load_data(e_path).reshape(-1, 2, 3)
    V_k        = load_data(V_k_path)
    V_k_coord  = process_V_k(V_k, e)
    potentials = load_data(potentials_path)


    with open(singularity_points_path, 'rb') as file:
        singularity_points = pickle.load(file)

    
    
    # 绘制某一个时刻的速度场和临界点
    # potential = potentials[0]
    # velocity = V_k_coord[0]
    # plot_velocity_fields_and_singularity_points(surface, singularity_points, potential, velocity, plot_type='Raw')

    # 绘制k个时刻的速度场和临界点,并保存为gif
    plot_velocity_fields_and_singularity_points_gif(surface, potentials, V_k_coord, singularity_points, velocity_fields_gif_path)

    # 绘制k个时刻的速度场、临界点以及真实临界点,并保存为gif
    # with open(true_singularity_points_path, 'rb') as file:
    #     true_singularity_points = pickle.load(file)
    # plot_velocity_fields_and_singularity_points_and_true_singularity_points_gif(surface, potentials, V_k_coord, singularity_points, true_singularity_points, velocity_fields_gif_path)
