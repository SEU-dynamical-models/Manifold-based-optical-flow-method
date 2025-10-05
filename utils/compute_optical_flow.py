# Calculating velocity fields from ECoG data using optical flow method
# Author: Xi Wang
# Date: April 25, 2024
# Email: 2308180834@qq.com
"""
This code uses variational methods to calculate optical flow fields from ECoG data using finite element discretization. Key steps include:
- Load surface and interpolated potential data from files.
- Use the finite element method to calculate the quantities of variation problems (a2 matrix, basis function gradient, orthogonal basis, integral term).
- A parallel approach is used to calculate the velocity field for multiple time steps.
- Save the velocity field and orthogonal basis to CSV files.
The code reads configuration parameters from a YAML file and utilizes multiple processes to speed up calculations.
"""
from multiprocessing import Pool
import os
import time
import numpy as np
import pandas as pd
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import multiprocessing

import yaml


def compute_geometrical_quantities(coordinates, normals, triangles, areas):
    '''
    计算几何量
    
    参数: 
        surface: 输入的曲面对象
            - coordinates: 顶点坐标数组(point_num, 3)
            - triangles: 三角形索引数组(mesh_num, 3)
            - normals: 顶点法向量数组(point_num, 3)
            - areas: 三角形面积数组(mesh_num, 1)
    
    返回值: 
        a2: a2系数矩阵 (2*point_num, 2*point_num)
        grad_w: wi梯度矩阵 (mesh_num, 3, 3)
        e: 正交基数组 (point_num, 2, 3)
        integral_wi_wj: 积分项数组 (mesh_num, 2),分为i=j和i!=j两种情况
    '''
    start = time.time()
    point_num = len(coordinates)
    mesh_num = len(triangles)

    # a2             = np.zeros((2*point_num, 2*point_num))  # 初始化 a2 矩阵
    a2 = lil_matrix((2 * point_num, 2 * point_num))  # 初始化 a2 矩阵(稀疏矩阵)
    grad_w = np.zeros((mesh_num, 3, 3))  # 初始化 grad_w 矩阵
    e = np.zeros((point_num, 2, 3))  # 初始化正交基数组
    integral_wi_wj = np.zeros((mesh_num, 2))  # 初始化积分项数组

    # 计算曲面上每一点切空间内的正交基底
    for i, coord in enumerate(coordinates):
        n_i = normals[i]
        e[i][0], e[i][1] = compute_orthonormal_basis(n_i)

    # 计算a2矩阵,grad_w矩阵,integral_wi_wj矩阵
    for index, tri in enumerate(triangles):
        # 计算grad_w[index]
        A, B, C = tri
        grad_w[index][0] = compute_gradient_w(coordinates[A], coordinates[B],
                                              coordinates[C])
        grad_w[index][1] = compute_gradient_w(coordinates[B], coordinates[A],
                                              coordinates[C])
        grad_w[index][2] = compute_gradient_w(coordinates[C], coordinates[A],
                                              coordinates[B])

        grad_w_index = grad_w[index]

        # 计算integral_wi_wj[index]
        A_T = areas[index]
        integral_wi_wj[index][0] = A_T / 6
        integral_wi_wj[index][1] = A_T / 12

        # 计算a2[i + point_num * alpha, j + point_num * beta]
        for i_order, i in enumerate(tri):
            for j_order, j in enumerate(tri):
                if i <= j:  # 只计算上三角部分
                    for alpha in range(2):
                        for beta in range(2):
                            a2[i + point_num * alpha,
                               j + point_num * beta] += compute_a2(
                                   A_T, e[i][alpha], e[j][beta],
                                   grad_w_index[i_order],
                                   grad_w_index[j_order])
                            if i != j:  # 对称位置赋值
                                a2[j + point_num * beta, i +
                                   point_num * alpha] = a2[i +
                                                           point_num * alpha,
                                                           j +
                                                           point_num * beta]
    end = time.time()
    execution_time = end - start

    return a2, grad_w, e, integral_wi_wj, execution_time


def worker(k, a2, grad_w, e, integral_wi_wj, triangles, t_k, areas, lambda_,
           I_k_k, I_k_kplus1):
    print(f"第{k}个速度场")
    point_num = len(e)
    # 初始化稀疏矩阵a1和向量f
    # a1 = np.zeros((2 * point_num, 2 * point_num))
    a1 = lil_matrix((2 * point_num, 2 * point_num))
    f = np.zeros(2 * point_num)
    # I_k_k = I_k[k]
    # I_k_kplus1 = I_k[k+1]
    # I_k_kplus1 = I_k_2[k]

    # 计算矩阵a1和向量f
    for index, T in enumerate(triangles):
        # 计算grad_M_I
        grad_w_index = grad_w[index]
        grad_M_I = I_k_k[T[0]] * grad_w_index[0] + I_k_k[
            T[1]] * grad_w_index[1] + I_k_k[T[2]] * grad_w_index[2]

        # 计算矩阵a1和向量f
        for i_order, i in enumerate(T):
            for alpha in range(2):
                # 计算f[i + point_num * alpha]
                f[i + point_num * alpha] += compute_f(grad_M_I, e[i][alpha],
                                                      I_k_kplus1, I_k_k,
                                                      t_k[k + 1] - t_k[k], i,
                                                      T, areas[index])
                for j_order, j in enumerate(T):
                    if i <= j:  # 只计算上三角部分
                        for beta in range(2):
                            # 计算a1[i + point_num * alpha, j + point_num * beta]
                            integral = integral_wi_wj[index][
                                0] if i == j else integral_wi_wj[index][1]
                            a1[i + point_num * alpha,
                               j + point_num * beta] += compute_a1(
                                   integral, grad_M_I, e[i][alpha], e[j][beta])
                            if i != j:  # 对称位置赋值
                                a1[j + point_num * beta, i +
                                   point_num * alpha] = a1[i +
                                                           point_num * alpha,
                                                           j +
                                                           point_num * beta]

    # 利用V = a^{-1}f求解V
    a = a1 + lambda_ * a2
    # print("矩阵a", a)
    a = csr_matrix(a)  # 将稀疏矩阵转换为CSR格式
    V = spsolve(a, f)  # 使用稀疏矩阵求解线性方程组
    # queue.put((k, V))
    return V


def compute_velocity_field(processes_num, time_steps, a2, grad_w, e,
                           integral_wi_wj, triangles, t_k, areas, lambda_, I_k,
                           I_k_2):
    V_k_processes = []
    V_k = []
    pool = Pool(processes_num)
    execution_time = 0
    try:
        start_time = time.time()

        for k in range(time_steps - 1):
            r = pool.apply_async(worker,
                                 args=(
                                     k,
                                     a2,
                                     grad_w,
                                     e,
                                     integral_wi_wj,
                                     triangles,
                                     t_k,
                                     areas,
                                     lambda_,
                                     I_k[k],
                                     I_k_2[k + 1],
                                 ))
            V_k_processes.append(r)
        pool.close()
        pool.join()

        end_time = time.time()
        execution_time = end_time - start_time
    finally:
        # print("程序执行时间：", execution_time, "秒")
        # print("结点个数：", len(surface.points))
        # print("三角形面片个数：", len(triangles))
        # print("平均每个速度场计算时间：", execution_time / (time_steps - 1), "秒")
        # print(results)

        for v in V_k_processes:
            V_k.append(v.get())

        # print(V_k)
    return V_k, execution_time


def load_surface(surface_path):
    # 读取表面数据
    surface = pv.read(surface_path)
    return surface


def load_potentials(csv_path):
    # 读取电势数据
    potentials = pd.read_csv(csv_path, sep=',', header='infer',
                             index_col=0).values
    return potentials


def compute_orthonormal_basis(n_i):
    '''
    计算法向量n_i对应的切空间中的正交基底
    
    参数: 
        n_i: 顶点法向量
    
    返回值: 
        e1: 正交基向量1
        e2: 正交基向量2
    '''
    # Compute an orthonormal basis e1(i), e2(i) given the normal vector n(i)
    if n_i[0] != 0 or n_i[1] != 0:
        e1 = np.array([-n_i[1], n_i[0], 0])
    else:
        e1 = np.array([0, -n_i[2], n_i[1]])

    # Step 3: Find another vector perpendicular to the normal vector and the first tangent vector
    e2 = np.cross(n_i, e1)

    # Normalize the tangent vectors
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    # Step 4: Return the two tangent vectors
    return e1, e2


def compute_gradient_w(p_i, p_j, p_k):
    '''
    计算梯度
    
    参数: 
        p_i, p_j, p_k: 顶点坐标
    
    返回值: 
        gradient_w: 梯度值
    '''
    # Compute the gradient of w_i on triangle T
    vector_jk = p_k - p_j
    vector_ji = p_i - p_j
    perpendicular_vector = np.dot(vector_ji, vector_jk) * vector_jk / np.dot(
        vector_jk, vector_jk)
    vector_ih = p_j - p_i + perpendicular_vector
    gradient_w = vector_ih / np.dot(vector_ih, vector_ih)
    return gradient_w


def compute_a2(T_area, e_i, e_j, grad_i, grad_j):
    '''
    根据equation16计算a2(W_alpha_i,W_beta_j): a2(i+point_num*alpha, j+point_num*beta)
    
    参数: 
        T_area: 三角形T的面积
        e_i、e_j: 正交基向量
        grad_i、grad_j: 梯度
    
    返回值: 
        term: a2(W_alpha_i,W_beta_j)
    '''
    return np.dot(e_i, e_j) * np.dot(grad_i, grad_j) * T_area


def compute_a1(integral, grad_M_I, e_i, e_j):
    '''
    根据equation15计算a1(W_alpha_i,W_beta_j): a1(i+point_num*alpha, j+point_num*beta)
    
    参数: 
        integral: 积分项
        grad_M_I: I在三角形面片M上的梯度
        e_i、e_j: 正交基向量
    
    返回值: 
        term: a1(W_alpha_i,W_beta_j)
    '''
    return np.dot(grad_M_I, e_i) * np.dot(grad_M_I, e_j) * integral


def compute_f(grad_M_I, e_i, I_kplus1, I_k, t, i, T, T_area):
    '''
    根据equation17计算f(W_alpha_i): f(i+point_num*alpha)
    
    参数: 
        grad_M_I: 梯度
        e_i: 正交基向量
        I_kplus1: t[k+1]时刻对应的所有I
        I_k: t[k]时刻对应的所有I
        t: 时间差t[k+1]-t[k]
        i: 曲面中的点i
        T: 三角形T的索引
        T_area: 三角形T的面积
    
    返回值: 
        term: f(W_alpha_i)
    '''
    T = set(T)
    other_point = T - {i}
    partial_i = (I_kplus1[i] - I_k[i]) / t
    partial_other_points = np.sum([(I_kplus1[x] - I_k[x]) / t
                                   for x in other_point])
    return np.dot(
        e_i, grad_M_I) * (2 * partial_i + partial_other_points) * T_area / 12


def reshape_and_save_data(data, file_path):
    if isinstance(data, list):
        data = np.array(data)
    # 重塑数据并保存为CSV文件
    reshaped_data = data.reshape(data.shape[0], -1)
    pd.DataFrame(reshaped_data).to_csv(file_path)
    print(f"{file_path}文件保存成功。")


if "__main__" == __name__:
    with open("./config/opticalflow.yaml", 'r', encoding='UTF-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data_params    = config['sub_08']
    general_params = config['general']


    surface_path           = data_params['surface_path']
    potentials_path        = data_params['potentials_path']
    file_path_e            = data_params['e_path']
    file_path_V_k          = data_params['V_k_path']
    processed_surface_path = data_params['processed_surface_path']
    time_steps             = general_params['time_steps']
    processes_num          = general_params['processes_num']  # 进程数
    lambda_                = general_params['lambda_']  # 0.1
    # potentials_path_2      = config['potentials_path_2']

    surface    = load_surface(surface_path)
    potentials = load_potentials(potentials_path)
    # potentials_2           = load_potentials(potentials_path_2)

    coordinates = surface.points
    triangles   = surface.faces.reshape(-1, 4)[:, 1:]
    point_num   = len(coordinates)
    mesh_num    = len(triangles)
    normals     = surface.point_normals
    areas       = surface.compute_cell_sizes(length=False, volume=False)['Area']

    # t_k                    = [i for i in range(num_tasks+1)]
    t_k = [i for i in range(time_steps)]
    I_k = potentials[t_k]
    # I_k_2 = potentials_2[t_k]

    print("结点个数：", len(surface.points))
    print("三角形面片个数：", len(triangles))

    a2, grad_w, e, integral_wi_wj, execution_time = compute_geometrical_quantities(
        coordinates, normals, triangles, areas)
    print("a2, grad_w, e, integral_wi_wj计算完成,花费时间为: ", execution_time)

    V_k, execution_time = compute_velocity_field(processes_num, time_steps, a2,
                                                 grad_w, e, integral_wi_wj,
                                                 triangles, t_k, areas,
                                                 lambda_, I_k, I_k)
    print("总的速度场计算时间：", execution_time, "秒")
    print("平均每个速度场计算时间：", execution_time / (time_steps - 1), "秒")

    # 保存数据到CSV文件中
    reshape_and_save_data(e, file_path_e)  # 保存正交基底e
    reshape_and_save_data(V_k, file_path_V_k)  # 保存速度场V_k
