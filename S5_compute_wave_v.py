import bz2
import json
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft
import pyvista as pv
import yaml
import final_compute_optical_flow
import final_draw_optical_flow_field
import sys
from scipy.signal import hilbert

def wave_velocity_amplitude(surface, potentials, dt, time_steps, e):
    """
    计算电势数据的波速（幅值法）
    """
    coordinates = surface.points
    triangles   = surface.faces.reshape(-1, 4)[:, 1:]
    point_num   = len(coordinates)
    areas       = surface.compute_cell_sizes(length=False, volume=False)['Area']

    # 时间导数
    time_derivative = np.gradient(potentials, axis=0, edge_order=2) / dt
    print(time_derivative.shape)

    # 空间梯度
    grad_point = compute_grad_M_I(coordinates, triangles, potentials, surface, areas)
    print(grad_point.shape)

    # 投影到切平面
    grad_point_proj = np.zeros((time_steps, point_num, 3))
    for t in range(time_steps):
        grad_point_now = grad_point[t]
        for index in range(point_num):
            grad_point_proj[t, index] = project_vector_to_plane(grad_point_now[index], e[index][0], e[index][1])
    print(grad_point_proj.shape)

    # 用正交基表示
    grad_point_proj_basis = np.zeros((time_steps, point_num, 2))
    for t in range(time_steps):
        grad_point_proj_now = grad_point_proj[t]
        for index in range(point_num):
            grad_point_proj_basis[t, index, 0], grad_point_proj_basis[t, index, 1] = express_vector_on_basis(grad_point_proj_now[index], e[index][0], e[index][1])
    print(grad_point_proj_basis.shape)

    # 计算模长
    grad_point_proj_basis_dis = np.zeros((time_steps, point_num))
    for t in range(time_steps):
        grad_point_proj_basis_now = grad_point_proj_basis[t]
        for index in range(point_num):
            grad_point_proj_basis_dis[t, index] = np.sqrt(grad_point_proj_basis_now[index, 0] ** 2 + grad_point_proj_basis_now[index, 1] ** 2)
    print(grad_point_proj_basis_dis.shape)

    # 波速 = 时间导数 / 空间梯度模长
    wave_velocity = time_derivative / grad_point_proj_basis_dis

    return wave_velocity

def compute_temporal_gradient_phase(data, dt):
    """
    计算相位数据的时间梯度（考虑相位环绕）
    """
    t = data.shape[0]
    gradients = np.zeros_like(data)

    # 第一个时间步，前向差分
    gradients[0] = angle_subtract(data[1], data[0], angleFlag=True) / dt

    # 中间时间步，中心差分
    for i in range(1, t-1):
        gradients[i] = angle_subtract(data[i+1], data[i-1], angleFlag=True) / (2 * dt)

    # 最后一个时间步，后向差分
    gradients[t-1] = angle_subtract(data[t-1], data[t-2], angleFlag=True) / dt

    return gradients

def wave_velocity_phase(surface, phases, dt, time_steps, e):
    """
    计算相位数据的波速（相位法）
    """
    coordinates = surface.points
    triangles   = surface.faces.reshape(-1, 4)[:, 1:]
    point_num   = len(coordinates)
    areas       = surface.compute_cell_sizes(length=False, volume=False)['Area']

    # 时间导数
    time_derivative = compute_temporal_gradient_phase(phases, dt)
    print(time_derivative.shape)

    # 空间梯度
    grad_point = compute_grad_M_I(coordinates, triangles, phases, surface, areas)
    print(grad_point.shape)

    # 投影到切平面
    grad_point_proj = np.zeros((time_steps, point_num, 3))
    for t in range(time_steps):
        grad_point_now = grad_point[t]
        for index in range(point_num):
            grad_point_proj[t, index] = project_vector_to_plane(grad_point_now[index], e[index][0], e[index][1])
    print(grad_point_proj.shape)

    # 用正交基表示
    grad_point_proj_basis = np.zeros((time_steps, point_num, 2))
    for t in range(time_steps):
        grad_point_proj_now = grad_point_proj[t]
        for index in range(point_num):
            grad_point_proj_basis[t, index, 0], grad_point_proj_basis[t, index, 1] = express_vector_on_basis(grad_point_proj_now[index], e[index][0], e[index][1])
    print(grad_point_proj_basis.shape)

    # 计算模长
    grad_point_proj_basis_dis = np.zeros((time_steps, point_num))
    for t in range(time_steps):
        grad_point_proj_basis_now = grad_point_proj_basis[t]
        for index in range(point_num):
            grad_point_proj_basis_dis[t, index] = np.sqrt(grad_point_proj_basis_now[index, 0] ** 2 + grad_point_proj_basis_now[index, 1] ** 2)
    print(grad_point_proj_basis_dis.shape)

    # 波速 = 时间导数 / 空间梯度模长
    wave_velocity = time_derivative / grad_point_proj_basis_dis

    return wave_velocity

def compute_gradient_w(p_i, p_j, p_k):
    '''
    计算三角形上某点的梯度
    '''
    vector_jk = p_k - p_j
    vector_ji = p_i - p_j
    perpendicular_vector = np.dot(vector_ji, vector_jk) * vector_jk / np.dot(vector_jk, vector_jk)
    vector_ih = p_j - p_i + perpendicular_vector
    gradient_w = vector_ih / np.dot(vector_ih, vector_ih)
    return gradient_w

def compute_grad_M_I(coordinates, triangles, potentials, surface, areas):
    """
    计算每个点的空间梯度
    """
    point_num = len(coordinates)
    mesh_num = len(triangles)
    time_steps = len(potentials)
    grad_w = np.zeros((mesh_num, 3, 3))  # 每个三角形三个顶点的梯度
    grad_M = np.zeros((time_steps, mesh_num, 3))  # 每个三角形的梯度
    grad_point = np.zeros((time_steps, point_num, 3))  # 每个点的梯度

    for index, tri in enumerate(triangles):
        A, B, C = tri
        grad_w[index][0] = compute_gradient_w(coordinates[A], coordinates[B], coordinates[C])
        grad_w[index][1] = compute_gradient_w(coordinates[B], coordinates[A], coordinates[C])
        grad_w[index][2] = compute_gradient_w(coordinates[C], coordinates[A], coordinates[B])
        grad_w_index = grad_w[index]
        for t in range(time_steps):
            grad_M_I = (
                potentials[t][tri[0]] * grad_w_index[0] +
                potentials[t][tri[1]] * grad_w_index[1] +
                potentials[t][tri[2]] * grad_w_index[2]
            )
            grad_M[t][index] = grad_M_I

    for index in range(point_num):
        ids = surface.point_cell_ids(index)
        all_areas = 0
        for id in ids:
            grad_point[:, index] += grad_M[:, id] * areas[id]
            all_areas += areas[id]
        if all_areas == 0:
            print("all_areas = 0", ids)
        grad_point[:, index] /= all_areas

    return grad_point

def project_vector_to_plane(V, e1, e2):
    """
    将向量V投影到由基底e1和e2所确定的平面上
    """
    n = np.cross(e1, e2)
    Vn = np.dot(V, n) * n / np.dot(n, n)
    Vt = V - Vn
    return Vt

def express_vector_on_basis(V, e1, e2):
    """
    表示向量V在由e1和e2构成的基底上的线性组合系数(alpha, beta)
    """
    V, e1, e2 = map(np.array, (V, e1, e2))
    if np.all(e1 == 0) or np.all(e2 == 0):
        raise ValueError("基底向量不能是零向量。")
    alpha = np.dot(V, e1) / np.dot(e1, e1)
    beta = np.dot(V, e2) / np.dot(e2, e2)
    return alpha, beta

def plot_loghist(x, bins):
    """
    绘制波速的对数直方图
    """
    plt.figure(figsize=(8, 6))
    plt.hist(x, bins=1500, density=True, alpha=0.6, color='blue', edgecolor='black', label='3-6Hz Wave')
    plt.xscale('log')
    plt.title('Wave Speeds')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

def plot_surface_wave_velocity(surface, wave_velocity):
    """
    绘制表面上的波速分布
    """
    p = pv.Plotter()
    p.add_mesh(surface, scalars=wave_velocity, cmap="jet", show_edges=False)
    p.show()

def compute_phase_from_potentials(potentials):
    """
    计算电势信号的瞬时相位
    """
    analytic_signal = hilbert(potentials)
    print(analytic_signal.shape)
    phases = np.angle(analytic_signal)
    print(phases.shape)
    return phases

@staticmethod
def angle_subtract(f1, f2, angleFlag=True):
    """
    角度相减，结果在[-pi, pi]区间
    """
    if angleFlag:
        fdiff = np.mod(f1 - f2 + np.pi, 2 * np.pi) - np.pi
    else:
        fdiff = f1 - f2
    return fdiff

def compute_orthonormal_basis(n_i):
    """
    计算法向量n_i对应的切空间中的正交基底
    """
    if n_i[0] != 0 or n_i[1] != 0:
        e1 = np.array([-n_i[1], n_i[0], 0])
    else:
        e1 = np.array([0, -n_i[2], n_i[1]])
    e2 = np.cross(n_i, e1)
    e1_ = e1 / np.linalg.norm(e1)
    e2_ = e2 / np.linalg.norm(e2)
    if np.sum(np.isnan(e1_)) > 0 or np.sum(np.isnan(e2_)) > 0:
        print(e1, e2, n_i)
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2

def reshape_and_save_data(data, file_path):
    """
    重塑数据并保存为CSV文件
    """
    if isinstance(data, list):
        data = np.array(data)
    reshaped_data = data.reshape(data.shape[0], -1)
    pd.DataFrame(reshaped_data).to_csv(file_path)
    print(f"{file_path}文件保存成功。")

if "__main__" == __name__:
    # 读取命令行参数
    subfolder  = sys.argv[1]
    run_num    = sys.argv[2]
    trial_name = sys.argv[3]

    # 路径设置
    data_path    = '/fred/oz284/mc/data/ds004080'
    results_path = '/fred/oz284/mc/results/ds004080'

    data_subfolder_path = f"{data_path}/{subfolder}"
    filelist            = os.listdir(data_subfolder_path)
    ses                 = filelist[0]

    surface_path = f"{results_path}/{subfolder}/{subfolder}_reconstructed_surface.ply"
    surface      = pv.read(surface_path)

    file_path_e     = f"{results_path}/{subfolder}/{subfolder}_e.csv"
    potentials_path = f"{results_path}/{subfolder}/run-{run_num}/{trial_name}/{subfolder}_{ses}_task-SPESclin_run-{run_num}-{trial_name}-ave-interpolation_phases_data.csv"
    json_data = f"{data_subfolder_path}/{ses}/ieeg/{subfolder}_{ses}_task-SPESclin_run-{run_num}_ieeg.json"

    coordinates = surface.points
    normals     = surface.point_normals

    point_num = len(coordinates)
    e         = np.zeros((point_num, 2, 3))  # 初始化正交基数组

    # 计算曲面上每一点切空间内的正交基底
    print(normals)
    for i, coord in enumerate(coordinates):
        n_i = normals[i]
        e[i][0], e[i][1] = compute_orthonormal_basis(n_i)
    print("nan: ", np.sum(np.isnan(e)))

    reshape_and_save_data(e, file_path_e)  # 保存正交基底e

    # 读取插值相位数据
    potentials = pd.read_csv(potentials_path, sep=',', header='infer', index_col=0).values

    # 读取采样率
    with open(json_data,'r',encoding='UTF-8') as f:
        json_info = json.load(f)
    SF = round(json_info["SamplingFrequency"])
    dt = 1 / SF

    time_steps = len(potentials)
    phases = potentials

    # 计算波速
    wave_velocity = wave_velocity_phase(surface, phases, dt, time_steps, e)
    wave_velocity /= 1000  # 转换单位为rad/ms
    wave_velocity = np.abs(wave_velocity)

    # 保存波速结果
    sl_fname = f"{results_path}/{subfolder}/run-{run_num}/{trial_name}/{subfolder}_{ses}_task-SPESclin_run-{run_num}-{trial_name}-wave_velocity.pkl.bz2"
    with bz2.BZ2File(sl_fname, 'wb') as file:
        pickle.dump(wave_velocity, file)