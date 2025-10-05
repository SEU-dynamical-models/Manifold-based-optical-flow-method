import math
import pickle
import pyvista as pv
import numpy as np
import yaml
from functools import partial

import final_draw_optical_flow_field
import sys
import bz2



# 计算法向量n_i对应切平面的基底
def compute_orthonormal_basis(n_i):
    """
    计算法向量n_i对应切平面的正交基底(e1, e2)。

    参数:
    - n_i (numpy.ndarray): 三维法向量

    返回:
    - numpy.ndarray: 切平面的基底向量e1
    - numpy.ndarray: 切平面的基底向量e2
    """
    if n_i[0] != 0 or n_i[1] != 0:
        e1 = np.array([-n_i[1], n_i[0], 0])
    else:
        e1 = np.array([0, -n_i[2], n_i[1]])

    e2 = np.cross(n_i, e1)

    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)

    return e1, e2
# 把向量V投影到基底为(e1, e2)的平面
def project_vector_to_plane(V, e1, e2):
    """
    将向量V投影到由基底e1和e2所确定的平面上。
    
    参数:
    - V (numpy.ndarray): 要投影的向量
    - e1 (numpy.ndarray): 平面的基底1
    - e2 (numpy.ndarray): 平面的基底2
    
    返回值:
    - (numpy.ndarray): 向量V在平面上的投影
    """
    # 计算平面的法向量
    n = np.cross(e1, e2)
    # 计算向量V在法向量方向上的分量
    Vn = np.dot(V, n) * n / np.dot(n, n)
    # 计算向量V在平面上的投影
    Vt = V - Vn
    return Vt

# 计算B点相对于A点, 在(e1, e2)方向上分别的位置差
def position_diff_on_basis_with_origin(A, B, e1, e2):
    """
    计算点B相对于点A在基底(e1, e2)定义的平面上的位置差。

    参数:
    - A (numpy.ndarray): 点A的坐标
    - B (numpy.ndarray): 点B的坐标
    - e1 (numpy.ndarray): 基底向量e1
    - e2 (numpy.ndarray): 基底向量e2

    返回:
    - float: 点B相对于点A在e1方向上的位置差
    - float: 点B相对于点A在e2方向上的位置差
    """

    # 将点B的坐标转换到以A为原点的坐标系
    B_relative = B - A

    # 计算B_relative在切平面上的投影向量
    n = np.cross(e1, e2)
    proj = B_relative - np.dot(B_relative, n) * n / np.dot(n, n)

    # 计算B_relative在基底坐标系下的坐标
    u = np.dot(proj, e1)
    v = np.dot(proj, e2)

    return u, v

# 把(e1, e2)平面上的向量V,用基底线性组合的方式表示
def express_vector_on_basis(V, e1, e2):
    """
    表示向量V在由e1和e2构成的基底上的线性组合。

    参数:
    - V (list): 需要表示的速度向量。
    - e1 (list): 基底向量e1。
    - e2 (list): 基底向量e2。

    返回:
    - tuple
        向量V在基底e1和e2上的线性组合系数(alpha, beta)。
    """

    # 将输入向量转换为NumPy数组
    V, e1, e2 = map(np.array, (V, e1, e2))

    # 确保基底向量不是零向量
    if np.all(e1 == 0) or np.all(e2 == 0):
        raise ValueError("基底向量不能是零向量。")

    alpha = np.dot(V, e1) / np.dot(e1, e1)
    beta = np.dot(V, e2) / np.dot(e2, e2)

    return alpha, beta

def angle_between_vectors(v1, v2):
    """
    计算两个向量v1和v2之间的有序角度差(以弧度为单位)v2-v1
    按照逆时针方向为正
    """
    v1_u = v1 / np.linalg.norm(v1)  # 单位向量
    v2_u = v2 / np.linalg.norm(v2)  # 单位向量
    
    dot_product = np.dot(v1_u, v2_u)  # 向量点积
    if dot_product > 1:
        dot_product = 1
    elif dot_product < -1:
        dot_product = -1
    angle = np.arccos(dot_product)  # 角度(弧度)
    
    # 计算有序角度差
    cross_product = np.cross(v1_u, v2_u)
    if cross_product < 0:
        angle = -angle  # 逆时针为正
    
    return angle

def winding_number(vx, vy):
    """
    计算2D矢量场(vx, vy)在封闭路径(x, y)上的绕圈数
    """
    n = len(vx)
    winding_number = 0
    for i in range(n):
        v1 = [vx[i], vy[i]]
        v2 = [vx[(i + 1) % n], vy[(i + 1)%n]]
        angles_diff = angle_between_vectors(v1, v2)
        winding_number += angles_diff
    return winding_number / (2 * np.pi)


def polar_angle(x, y, cx, cy):
    """计算点(x, y)相对于中心点(cx, cy)的极坐标角度"""
    return np.arctan2(y - cy, x - cx)

def sort_by_polar_angle_anticlockwise(point, x, y, vx=None, vy=None):
    """根据极坐标角度对点进行逆时针排序"""
    n = len(x)
    cx, cy = point
    # angles = polar_angle(x, y, cx, cy)
    # distances = (x - cx)**2 + (y - cy)**2
    # sorted_indices = np.lexsort((-angles, distances))

    values = [math.atan2(y[i], x[i]) for i in range(n)]
    sorted_indices = np.lexsort((values, ))

    if vx is not None and vy is not None:
        return sorted_indices, x[sorted_indices], y[sorted_indices], vx[sorted_indices], vy[sorted_indices]
    else:
        return sorted_indices, x[sorted_indices], y[sorted_indices]
    

def check_property(data, flag):
    """
    判断绕圈数是否满足奇点性质

    参数：
    data (float)：绕圈数数据
    flag (int)：标志，表示奇点类型（1为节点，-1为鞍）

    返回：
     (bool)：表示是否满足奇点性质

    """
    if flag == 1:
        if data >= 0.999 and data <= 1.001:
            return True
        else:
            return False
    elif flag == -1:
        if data >= -1.001 and data <= -0.999:
            return True
        else:
            return False
        

def calculate_winding_numbers(surf, singularity_points, V_now, e, points, max_level=25):
    """
    为曲面上当前时刻的所有奇点计算符合要求的绕圈数个数

    参数：
    surf (pyvista.PolyData)：表示曲面的PyVista PolyData对象
    singularity_points (list)：包含奇点坐标的列表
    V_now (list)：当前时刻的速度场
    e (list)：曲面上每个点切空间的基底
    points (numpy.ndarray)：表示曲面上所有点坐标的NumPy数组
    max_level (int, optional)：迭代的最大级别，默认为50

    返回：
    winding_numbers_counts (list)：包含每个奇点的符合要求的绕圈数计数的列表
    types (list)：包含每个奇点类型的列表（1为节点，-1为鞍）

    """
    winding_numbers_counts = []  # 创建一个空列表，用于存储每个奇点符合要求的绕圈数计数
    types = []  # 存储每个奇点的类型（1为节点，-1为鞍）

    for singularity_point in singularity_points:
        winding_numbers = []
        winding_numbers_count = 0
        index = surf.find_closest_point(singularity_point)
        point_neighbors_levels = surf.point_neighbors_levels(index, max_level)
        point_neighbors_levels = list(point_neighbors_levels)
        flag = 0

        for level in range(max_level):
            point_neighbors = point_neighbors_levels[level]
            e1, e2 = e[index]

            position_diff = [position_diff_on_basis_with_origin(points[index], points[x], e1, e2) for x in point_neighbors]
            position_diff = np.array(position_diff)
            x = position_diff[:, 0]
            y = position_diff[:, 1]

            V_point_neighbors = [V_now[x] for x in point_neighbors]
            V_point_neighbors_proj = [project_vector_to_plane(v, e1, e2) for v in V_point_neighbors]
            Vxy = [express_vector_on_basis(v, e1, e2) for v in V_point_neighbors_proj]
            Vxy = np.array(Vxy)
            Vx = Vxy[:, 0]
            Vy = Vxy[:, 1]

            # 按逆时针极角对坐标和向量进行排序
            sorted_indices, sorted_x, sorted_y, sorted_vx, sorted_vy = sort_by_polar_angle_anticlockwise((0, 0), x, y, Vx, Vy)
            winding_numbers.append(winding_number(sorted_vx, sorted_vy))

            if level == 0:
                if -1.01 <= winding_numbers[level] <= -0.99:
                    flag = -1
                    winding_numbers_count += 1
                    types.append(-1)
                elif 0.99 <= winding_numbers[level] <= 1.01:
                    flag = 1
                    winding_numbers_count += 1
                    types.append(1)
            else:
                if check_property(winding_numbers[level], flag):
                    winding_numbers_count += 1
                else:
                    break

        winding_numbers_counts.append(winding_numbers_count)

    return winding_numbers_counts, types

def calculate_scale_values(surf, singularity_points, winding_numbers_counts, points):
    """
    计算scale值映射函数

    参数：
    surf (pyvista.PolyData)：表示曲面的PyVista PolyData对象
    singularity_points (list)：包含奇点坐标的列表
    winding_numbers_counts (list)：包含每个奇点的绕圈数计数的列表
    points (numpy.ndarray)：表示曲面上所有点坐标的NumPy数组

    返回：
    scale (numpy.ndarray)：包含每个点的scale值映射的NumPy数组
    all_points (list)：用于可视化显示

    """
    scale = np.zeros(len(points))
    all_points = []

    for idx, singularity in enumerate(singularity_points):
        index = surf.find_closest_point(singularity)  # 找到离奇点最近的曲面上的点的索引
        scale[index] = winding_numbers_counts[idx] + 1

        point_neighbors_levels = surf.point_neighbors_levels(index, winding_numbers_counts[idx])
        point_neighbors_levels = list(point_neighbors_levels)   # 获取离奇点最近的相邻点索引列表

        for i in range(winding_numbers_counts[idx]):
            nei = point_neighbors_levels[i]
            for x in nei:
                scale[x] = winding_numbers_counts[idx] - i
                all_points.append(points[x])

    return scale, all_points

# def visualize_results(surf, winding_numbers_counts, types, all_points):
#     """
#     可视化结果函数

#     参数：
#     surf (pyvista.PolyData)：表示曲面的PyVista PolyData对象
#     winding_numbers_counts (list)：包含每个奇点的绕圈数计数的列表
#     types (list)：包含每个奇点类型的列表（1为节点，-1为鞍）
#     all_points (list)：包含所有点的列表，用于可视化显示

#     返回：
#     无

#     """
#     surf['scale'] = scale
#     max_scale = max(winding_numbers_counts)
#     contours = surf.contour(isosurfaces=np.arange(1.0, float(max_scale) + 1), scalars='scale') # 创建等值面对象contours，使用从1.0到最大比例尺的等间隔值作为等值面的标量值，并使用'scale'标量进行计算
#     p = pv.Plotter()
#     p.add_mesh(contours)
#     p.add_mesh(surf, show_edges=False)
#     # p.add_mesh(pv.PolyData(all_points), color='red', point_size=10)
#     p.show()

if "__main__" == __name__:
    with open("config.yaml", 'r', encoding='UTF-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    data_params = config['sub_01']
    general_params = config['general']
    surface_path = data_params['surface_path']


    potentials_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-ave-interpolation_data.csv"
    e_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-e.csv"
    V_k_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-V_k.csv"
    singularity_points_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-singularity_points.pkl"


    trial_name = sys.argv[1]

    e          = final_draw_optical_flow_field.load_data(e_path).reshape(-1, 2, 3)
    V_k        = final_draw_optical_flow_field.load_data(V_k_path)
    V_k_coord  = final_draw_optical_flow_field.process_V_k(V_k, e)
    potentials = final_draw_optical_flow_field.load_data(potentials_path)


    with open(singularity_points_path, 'rb') as file:
        singularity_points = pickle.load(file)


    surf    = pv.read(surface_path)
    normals = surf.point_normals
    points  = surf.points
    # print(len(points))

    sum = 0
    for x in singularity_points:
        if len(x) == 0:
            sum += 1
            continue
        print(f"{sum}, {len(x)}")
        sum += 1

    winding_lines = {}

    time_index = len(singularity_points)
    for t in range(time_index):
        winding_line = []
        singularity_point = singularity_points[t]
        V_now = V_k_coord[t]
        if len(singularity_point) == 0:
            continue
        winding_numbers_counts, types = calculate_winding_numbers(surf, singularity_point, V_now, e, points)
        print(t, winding_numbers_counts, types)
        num = len(winding_numbers_counts)

        sum = 0

        for x in range(num):
            if (winding_numbers_counts[x] == 0):
                continue
            tmp = []
            tmp.append(singularity_point[x])
            tmp.append(winding_numbers_counts[x])
            tmp.append(types[sum])
            winding_line.append(tmp)
            sum += 1
        winding_lines[str(t)] = winding_line
    
    sl_fname = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-winding_lines.pkl.bz2"
    with bz2.BZ2File(sl_fname, 'wb') as file:
        pickle.dump(winding_lines, file)

    # time_index = 0
    # V_now = V_k_coord[time_index]
    # singularity_points = singularity_points[time_index]

    # print(f"singularity_points:{singularity_points}")

    # winding_numbers_counts, types = calculate_winding_numbers(surf, singularity_points, V_now, e, points)
    # print(winding_numbers_counts, types)
    # # scale, all_points = calculate_scale_values(surf, singularity_points, winding_numbers_counts, points)
    # # visualize_results(surf, winding_numbers_counts, types, all_points)

