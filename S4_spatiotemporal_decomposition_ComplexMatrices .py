# Spatiotemporal mode decomposition of velocity field
# Author: Xi Wang
# Date: May 5, 2024
# Email: 2308180834@qq.com
"""
This code performs a spatiotemporal pattern decomposition of the velocity field.
The velocity data are first processed, converted into a representation in complex form, and then singular value decomposition (SVD) is applied to extract the main spatiotemporal patterns.
The variance percentage contribution of different modes is then calculated and the extracted modes are visualized by plotting the corresponding velocity fields.
"""
from matplotlib import pyplot as plt
import numpy as np
from utils import draw_optical_flow_field
import yaml
import pyvista as pv




def process_V_k_to_complex(V_k):
    """
    将V_k转换为复数形式的坐标表示。

    参数：
    - V_k (list): 速度分量列表, 包含每个点在基底方向的速度分量 [V1_1, V2_1, ..., Vn_1, V1_2, V2_2, ..., Vn_2]。
    - e (list): 基底列表, 包含每个点的基底。

    返回：
    - V_k_coord_complex (numpy.ndarray): 复数形式的速度坐标表示, 每个元素包含每个点在基底方向上的速度矢量的复数表示[V1_1+V1_2, V2_1+V2_2j, ..., Vn_1+Vn_2j]。
    """
    point_num = len(V_k[0]) // 2 # 曲面上点的个数
    V_k_array = np.zeros((len(V_k), point_num, 2))

    for k, V_index in enumerate(V_k):
        V = []
        for i in range(point_num):
            # 构建每个点在基底方向的速度分量 [V_x, V_y]
            V_k_array[k][i] = [V_index[i], V_index[i + point_num]]
    # print(V_k_array.shape)

    V_k_coord_complex = np.empty((len(V_k_array), point_num), dtype=complex)
    for k, V_index in enumerate(V_k_array):
        for i, v in enumerate(V_index):
            # 计算每个点在基底方向上的速度矢量
            V_k_coord_complex[k][i] = complex(v[0], v[1])

    return V_k_coord_complex


def calculate_V_k_from_complex(V_k_coord_complex, e):
    """
    从复数形式的速度坐标表示计算每个点的速度矢量。

    参数：
    - V_k_coord_complex (numpy.ndarray): 复数形式的速度坐标表示, 每个元素包含每个点在基底方向上的速度矢量的复数表示 [V1_1+V1_2, V2_1+V2_2j, ..., Vn_1+Vn_2j]。
    - e (list): 基底列表。

    返回：
    - V_k_coord (list): 每个点的速度矢量。
    """
    V_k_coord = []
    for i, v in enumerate(V_k_coord_complex):
        # 计算每个点在基底方向上的速度矢量
        V_1 = v.real * e[i][0]
        V_2 = v.imag * e[i][1]
        # if i == 0:
        #     print(V_index[i][0], e[i][0])
        #     print(V_1, V_2)
        V_k_coord.append(V_1 + V_2)
    return V_k_coord


def plot_surface_with_velocity_arrows(surface_path, velocity, type='Raw'):
    """
    绘制速度场。

    参数：
    - surface_path (string): 曲面数据文件的路径。
    - velocity (list): 曲面上每个点的速度向量。
    - type (string): 可选参数, 绘制类型。默认为 'Raw', 还可以选择 'Scaled'。

    返回：
    无返回值, 直接显示绘制的速度场。

    """
    surface               = pv.read(surface_path)  # 读取曲面数据
    surface['V']          = velocity  # 设置曲面上每个点的速度向量

    if type == 'Raw':
        # 根据速度向量的模长设置箭头的缩放比例scale
        lengths            = np.linalg.norm(velocity, axis=1)
        max_value          = np.max(lengths)
        scale_factor       = 10 / max_value
        scaled_lengths     = lengths * scale_factor
        surface['V_scale'] = scaled_lengths

        args               = {
            'tolerance': 0.01,
            # 'factor'   : 0.005,
            'factor': 0.2,
            'scale'    : 'V_scale',
            'orient'   : "V"
        }
    elif type == 'Scaled':
        # scale设置为相同的数值,速度场仅显示方向,不体现变化大小
        surface['V_scale'] = np.ones((len(velocity), )) * 5

        args = {
            'tolerance': 0.01,
            # 'factor': 0.005,
            'factor': 0.7,
            'scale': 'V_scale',
            'orient': "V"
        }
    else:
        print("Wrong Type!")

    p = pv.Plotter()
    # p.add_text("original", font_size=15)
    arrows = surface.glyph(**args)
    p.add_mesh(arrows, color="black")
    p.add_mesh(surface, color="grey", opacity=0.5, show_edges=False, smooth_shading=False)

    p.show()


def calculate_percentages(Sigma):
    """
    计算不同模式的占比, Sigma为奇异值。

    参数：
    - Sigma (numpy.ndarray): 一维数组, 表示奇异值。

    返回：
    - percentages (numpy.ndarray): 一维数组, 表示 Sigma 的每个元素相对于总和的百分比。
    - percentages_squared (numpy.ndarray): 一维数组, 表示 Sigma 的每个元素平方相对于总平方和的百分比。
    """
    squared_Sigma = np.square(Sigma)
    sum_of_squared_Sigma = np.sum(squared_Sigma)
    sum_of_Sigma = np.sum(Sigma)
    percentages_squared = (squared_Sigma / sum_of_squared_Sigma) * 100
    percentages = (Sigma / sum_of_Sigma) * 100
    return np.round(percentages, 2), np.round(percentages_squared, 2)


def extract_modes(Sigma, VT, e, k):
    """
    提取前 k 个主模式。

    参数：
    - Sigma (numpy.ndarray): 一维数组, 表示不同模式的奇异值。
    - VT (numpy.ndarray): 表示不同模式对应的投影系数。
    - e (list): 基底列表。
    - k (int): 表示要提取的主模式的数量。

    返回：
    无返回值, 直接调用plot_surface_with_velocity_arrows函数绘制提取出的模式的速度场。
    """
    for t in range(k):
        sigma = Sigma[t]
        vt = VT[t, :]
        V_k_decomposition = calculate_V_k_from_complex(sigma * vt, e)
        plot_surface_with_velocity_arrows(surface_path, V_k_decomposition, "Scaled")

if "__main__" == __name__:
    # 读取配置文件
    with open("code/ccep/opticalflow.yaml", 'r', encoding='UTF-8') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    data_params = config['sub_01']

    surface_path                           = data_params['surface_path']
    potentials_path                        = data_params['potentials_path']
    e_path                                 = data_params['e_path']
    V_k_path                               = data_params['V_k_path']
    singularity_points_path                = data_params['singularity_points_path']
    # singularity_points_classification_path = data_params['singularity_points_classification_path']


    # 处理数据
    e          = draw_optical_flow_field.load_data(e_path).reshape(-1, 2, 3)
    V_k        = draw_optical_flow_field.load_data(V_k_path)
    V_k_coord  = process_V_k_to_complex(V_k)
    potentials = draw_optical_flow_field.load_data(potentials_path)


    # SVD分解
    U, Sigma, VT = np.linalg.svd(V_k_coord, full_matrices=1)
    # print(U.shape, Sigma.shape, VT.shape)
    # print(Sigma)
    

    # 计算模式占比
    rounded_percentages, rounded_percentages_squared = calculate_percentages(Sigma)
    print(rounded_percentages)
    print(rounded_percentages_squared)


    # # 提取主模式
    # k = 4  # 保留前 4 个主模式
    # extract_modes(Sigma, VT, e, k)

    nmodeplot = 4
    U = U[:,:nmodeplot]
    Sigma = Sigma[:nmodeplot]
    VT = VT[:nmodeplot,:]
    isNegative = np.mean(np.real(U), axis=0) < 0

    # print(U.shape)
    # print(VT.shape)
    # print(isNegative)
    U[:, isNegative] = -U[:, isNegative]

    VT[isNegative, :] = -VT[isNegative, :]

    
    nt = len(V_k_coord)
    # Find average trajectory across all trials
    re_uav = np.zeros((nt, nmodeplot))
    im_uav = np.zeros((nt, nmodeplot))
    abs_uav = np.zeros((nt, nmodeplot))
    for it in range(nt):
        uav = U[it,:nmodeplot]
        re_uav[it,:] = np.real(uav)
        im_uav[it,:] = np.imag(uav)
        abs_uav[it,:] = np.abs(uav)
    uav_lims = [np.min([re_uav.ravel(), im_uav.ravel(), abs_uav.ravel()]),
                    np.max([re_uav.ravel(), im_uav.ravel(), abs_uav.ravel()])]
    

    # 创建图形对象
    fig, ax = plt.subplots(figsize=(8, 6))
    realTime = np.linspace(0.009, 0.2, nt)
    colors=['blue','green','orange', 'red']
    for imode in range(nmodeplot):
        # ax.plot(realTime, re_uav[:, imode], color='blue', label='re_uav')
        # ax.plot(realTime, im_uav[:, imode], color='green', label='im_uav')
        ax.plot(realTime, abs_uav[:, imode], color=colors[imode], label=f"Mode {imode + 1}")

    ax.set_ylim(uav_lims)
    ax.set_xlim([realTime[0], realTime[-1]])
    # ax.legend(prop={'size': 18})
    plt.rc("legend", fontsize=10)
    ax.axvline(0, color='black', linewidth=2.5)
    ax.axhline(0, color='black')
    ax.set_xlabel('Time(s)', fontsize=14)
    ax.set_ylabel('Absolute Value', fontsize=14)  # 'Absolute Value' 'Real Part' 'Imaginary Part'
    ax.set_title('SVD Modes Evolution', fontsize=16)
    ax.legend()  # 添加图例

    # 显示图形
    plt.show()