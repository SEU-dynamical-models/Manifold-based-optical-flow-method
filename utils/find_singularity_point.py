# Singularity detection in velocity field
# Author: Xi Wang
# Date: April 27, 2024
# Email: 2308180834@qq.com
"""
This code is designed to detect singularities in the velocity field.
First load the velocity field and orthogonal basis from csv files, and then identify all singularities in the velocity field.
It computes the Jacobian matrix for each singularity and classifies singularities based on the trace and determinant of the Jacobian matrix.
"""
import json
import pickle
import statistics
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
import pandas as pd
import yaml

# V_k (3, 6202)
# e (3101, 6)


# 加载CSV文件数据
def load_data(file_path):
    return pd.read_csv(file_path, header='infer', index_col=0).values

# 处理V_k,返回每个点在切平面上速度矢量的列表
def process_V_k(V_k, e):
    """
    将给定的 V_k 和 e 进行处理，并返回 V_k_coord。

    参数:
    - V_k (list): 包含曲面上每个点的速度分量的列表，每个元素是一个包含 e1 和 e2 方向分量的列表。
    - e (list): 包含曲面上每个点的基底方向的列表，每个元素是一个包含 e1 和 e2 基向量的列表。

    返回:
    - V_k_coord (list): 包含每个点在切平面上速度矢量的列表。

    """

    point_num = len(e)  # 曲面上点的个数
    V_k_array = []

    for k in range(len(V_k)):
        V = []
        V_index = V_k[k]
        for i in range(point_num):
            # 构建每个点在基底方向的速度分量 [V_x, V_y]
            V.append([V_index[i], V_index[i + point_num]])
        V_k_array.append(V)
        # V = np.array(V)
    V_k_array = np.array(V_k_array)
    print(V_k_array.shape)

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

# 判断顶点处速度是否为0
def is_zero_velocity_vertex(V_index, eps):
    """
    检查速度向量是否为0向量(或接近0)。

    参数:
    - V_index (list): 待检查的速度向量
    - eps (float): 用于判断速度是否接近零的阈值

    返回:
    - bool: 如果顶点处速度为0(或接近0),则返回 True,否则返回 False。

    """

    # 判断三个顶点处是否存在速度为0的点
    if np.linalg.norm(V_index) <= eps:
        return True
    # if np.allclose(V_index, np.zeros(3), atol=atol):
    #     return True
    return False

# 判断三角形ABC内部是否存在速度为0的点
def has_zero_velocity_interior(A, VA, B, VB, C, VC):
    """
    判断三角形ABC内部是否存在速度为零的点。

    参数:
    - A (numpy.ndarray): 三角形顶点A的坐标。
    - VA (numpy.ndarray): 三角形顶点A处的速度向量。
    - B (numpy.ndarray): 三角形顶点B的坐标。
    - VB (numpy.ndarray): 三角形顶点B处的速度向量。
    - C (numpy.ndarray): 三角形顶点C的坐标。
    - VC (numpy.ndarray): 三角形顶点C处的速度向量。

    返回:
    - bool: 如果三角形内部存在速度为0的点, 则返回True, 否则返回False
    - float: 速度为0的点在三角形内的重心坐标系中的λ值
    - float: 速度为0的点在三角形内的重心坐标系中的μ值

    """

    # 计算三角形所在平面的法向量
    n = np.cross(B - A, C - A)
    n /= np.linalg.norm(n)

    # 对速度向量进行投影
    VA_proj = VA - np.dot(VA, n) * n
    VB_proj = VB - np.dot(VB, n) * n
    VC_proj = VC - np.dot(VC, n) * n

    # 构造线性方程组
    M = np.column_stack((VA_proj - VC_proj, VB_proj - VC_proj))
    # print(VA_proj, VB_proj, VC_proj)

    # 求解线性方程组
    try:
        # print("hello")
        lam, mu = np.linalg.lstsq(M, -VC_proj, rcond=None)[0]
        # print(lam, mu)
        if lam + mu <= 1 and lam >= 0 and mu >= 0:
            # print(lam, mu)
            return [True, lam, mu]
        else:
            return [False, 0, 0]
    except np.linalg.LinAlgError:
        # 方程无解,不存在速度为0的点
        return [False, 0, 0]

# 找出网格中速度为0的顶点和内部点。
def find_singularity_points(coordinates, triangles, V_now, eps):
    """
    找出网格中速度为0的顶点和内部点。
    
    参数:
    coordinates (list): 网格中所有点的坐标
    triangles (list): 网格中所有三角形的顶点索引
    V_now (list): 每个点的当前速度向量
    eps (float): 判断速度为0的阈值
    
    返回值:
    singularity_vertices (list): 包含速度为0的顶点信息的列表
    singularity_interiors (list): 包含速度为0的内部点信息的列表
    v_length_max (float): 所有速度向量的最大长度
    """

    singularity_vertices = []
    singularity_interiors = []
    point_num = len(coordinates)

    # 计算每个速度向量的长度
    V_length = [np.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in V_now]
    v_length_max = np.max(V_length)

    # 检查顶点是否存在速度为0的点
    for i in range(point_num):
        if is_zero_velocity_vertex(V_now[i] / v_length_max, eps) is True:
            singularity_vertices.append([i, coordinates[i]])

    # 检查三角形内部是否存在速度为0的点
    for i, triangle in enumerate(triangles):
        if any(p in triangle for p, _ in singularity_vertices):
            continue

        A, B, C = triangle
        # VA, VB, VC = vels
        VA, VB, VC = V_now[A], V_now[B], V_now[C]
        flag, lam, mu = has_zero_velocity_interior(
            coordinates[A], VA / v_length_max, coordinates[B],
            VB / v_length_max, coordinates[C], VC / v_length_max)
        if flag:
            P_coord = lam * coordinates[A] + mu * coordinates[B] + (
                1 - lam - mu) * coordinates[C]
            singularity_interiors.append([
                i, P_coord, triangle, [lam, mu, 1 - lam - mu],
                [coordinates[A], coordinates[B], coordinates[C]]
            ])
            # singularity_interiors.append([index, triangle, [lam, mu]])

    return singularity_vertices, singularity_interiors, v_length_max


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


# 计算三角形(p1, p2, p3)平面上的法向量
def calculate_normal(p1, p2, p3):
    """
    计算三角形(p1, p2, p3)所在平面的法向量。

    参数:
    - p1 (numpy.ndarray): 三角形第一个顶点的三维坐标
    - p2 (numpy.ndarray): 三角形第二个顶点的三维坐标
    - p3 (numpy.ndarray): 三角形第三个顶点的三维坐标

    返回:
    - numpy.ndarray: 三角形所在平面的法向量
    """
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    return normal


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


# 找三角形(A, B, C)中的点P距离最近的边
def find_nearest_edge_and_vertices(A_coord, B_coord, C_coord, P_coord):
    """
    找到三角形(A, B, C)中距离点P最近的边,并返回该边的顶点坐标。

    参数:
    - A_coord (numpy.ndarray): 三角形顶点A的三维坐标
    - B_coord (numpy.ndarray): 三角形顶点B的三维坐标
    - C_coord (numpy.ndarray): 三角形顶点C的三维坐标
    - P_coord (numpy.ndarray): 待检测的点的三维坐标

    返回:
    - numpy.ndarray: 距离点P最近的边的两个顶点坐标
    """
    # 计算三角形三条边的向量
    v1 = B_coord - A_coord
    v2 = C_coord - B_coord
    v3 = A_coord - C_coord

    # 计算点P到三条边的距离
    d1 = np.abs(np.cross(P_coord - A_coord, v1) / np.linalg.norm(v1))
    d2 = np.abs(np.cross(P_coord - B_coord, v2) / np.linalg.norm(v2))
    d3 = np.abs(np.cross(P_coord - C_coord, v3) / np.linalg.norm(v3))

    # 找到距离最小的边
    distances = np.array([d1, d2, d3])
    nearest_edge_index = np.argmin(distances)

    # 返回距离点P最近的边的两个顶点坐标
    if nearest_edge_index == 0:
        return np.array([A_coord, B_coord])
    elif nearest_edge_index == 1:
        return np.array([B_coord, C_coord])
    else:
        return np.array([C_coord, A_coord])


# 为位于顶点的临界点计算雅可比矩阵
def compute_jacobian_matrix_for_vertex(singularity_vertice, V_now, surface,
                                       v_length_max, e):
    """
    计算特征顶点处的雅可比矩阵。

    参数:
    - singularity_vertice (list): 临界点信息列表
    - V_now (numpy.ndarray): 当前时刻的速度向量场
    - surface (object): 三角形网格表面对象
    - v_length_max (float): 当前时刻速度向量场中的最大向量长度

    返回:
    - numpy.ndarray: 特征顶点处的2x2雅可比矩阵
    """
    # 获取三角形网格的顶点坐标和面信息
    coordinates = surface.points
    triangles = surface.faces.reshape((-1, 4))[:, 1:]

    # 获取临界点的索引和相邻顶点
    index = singularity_vertice[0]
    near_points = surface.point_neighbors(index)
    # print(near_points)

    jacobian_matrix = np.zeros((2, 2))

    e1, e2 = e[index]

    # 遍历相邻顶点,计算雅可比矩阵
    for neighbor_index in near_points:
        # 获取相邻顶点的速度向量,并对其进行归一化
        # V_neighbor = V_now[neighbor_index] / np.linalg.norm(V_now[neighbor_index])
        # V_neighbor = V_now[neighbor_index]
        V_neighbor = V_now[neighbor_index] / v_length_max
        # 将速度向量投影到切平面上
        V_neighbor_projection = project_vector_to_plane(V_neighbor, e1, e2)
        # 将投影向量表示为基底(e1, e2)的线性组合
        u, v = express_vector_on_basis(V_neighbor_projection, e1, e2)
        # 计算相邻顶点相对于临界点在基底(e1, e2)上的位置差
        delta_e1, delta_e2 = position_diff_on_basis_with_origin(
            coordinates[index], coordinates[neighbor_index], e1, e2)
        # 更新雅可比矩阵的元素
        jacobian_matrix[0][0] += u / delta_e1
        jacobian_matrix[0][1] += u / delta_e2
        jacobian_matrix[1][0] += v / delta_e1
        jacobian_matrix[1][1] += v / delta_e2

    # print(jacobian_matrix)
    return jacobian_matrix

# 为位于三角形内部的临界点计算雅可比矩阵
def compute_jacobian_matrix_for_interior(singularity_interior, V_now, surface,
                                         v_length_max):
    """
    为位于三角形内部的临界点计算雅可比矩阵。

    参数:
    - singularity_interior (list): 临界点信息列表
    - V_now (numpy.ndarray): 当前时刻的速度向量场
    - surface (object): 三角形网格表面对象
    - v_length_max (float): 当前时刻速度向量场中的最大向量长度

    返回:
    - numpy.ndarray: 临界点处的2x2雅可比矩阵
    """
    coordinates = surface.points
    triangles = surface.faces.reshape((-1, 4))[:, 1:]

    # 从singularity_interior中提取所需的信息
    index, P_coord, tri, _, coords = singularity_interior
    A_coord, B_coord, C_coord = coords

    # 计算三角形平面的法向量,并构建正交基底
    normal = calculate_normal(A_coord, B_coord, C_coord)
    e1, e2 = compute_orthonormal_basis(normal)
    # P_coord = A_coord * lam + B_coord * mu + C_coord * gamma # P点坐标

    # 找到离临界点最近的边,并获取与该边相关的三角形
    nearest_edge_vertices = find_nearest_edge_and_vertices(
        A_coord, B_coord, C_coord, P_coord)
    triangle_id = (set(
        surface.find_cells_intersecting_line(nearest_edge_vertices[0],
                                             nearest_edge_vertices[1])) -
                   {index}).pop()
    near_points = list(set(tri) | set(triangles[triangle_id]))

    # 遍历相关顶点,计算雅可比矩阵
    jacobian_matrix = np.zeros((2, 2))
    for neighbor_index in near_points:
        # 获取相邻顶点的速度向量,并对其进行归一化
        # V_neighbor = V_now[neighbor_index] / np.linalg.norm(V_now[neighbor_index])
        # V_neighbor = V_now[neighbor_index]
        V_neighbor = V_now[neighbor_index] / v_length_max
        # 将速度向量投影到切平面上
        V_neighbor_projection = project_vector_to_plane(V_neighbor, e1, e2)
        # 将投影向量表示为基底(e1, e2)的线性组合
        u, v = express_vector_on_basis(V_neighbor_projection, e1, e2)
        # 计算相邻顶点相对于临界点在基底(e1, e2)上的位置差
        delta_e1, delta_e2 = position_diff_on_basis_with_origin(
            P_coord, coordinates[neighbor_index], e1, e2)
        # 更新雅可比矩阵的元素
        jacobian_matrix[0][0] += u / delta_e1
        jacobian_matrix[0][1] += u / delta_e2
        jacobian_matrix[1][0] += v / delta_e1
        jacobian_matrix[1][1] += v / delta_e2

    return jacobian_matrix

# 根据雅可比矩阵的迹和行列式,对临界点进行分类
def classify_critical_point(jacobian_matrix):
    """
    根据雅可比矩阵的特征值,对临界点进行分类。

    参数:
    - jacobian_matrix (numpy.ndarray): 2x2的雅可比矩阵

    返回:
    - str: 临界点的类型
    """
    # 计算雅可比矩阵的迹和行列式
    trace = np.trace(jacobian_matrix)
    determinant = np.linalg.det(jacobian_matrix)

    if determinant > 0:
        # 节点或焦点
        if trace**2 > 4 * determinant:
            return "Node"
            # 节点
            if trace > 0:
                return "Unstable Node"
            else:
                return "Stable Node"
        else:
            return "Focus"
            # 焦点
            if trace > 0:
                return "Unstable Spiral"
            else:
                return "Stable Spiral"
    elif determinant < 0:
        # 鞍点
        return "Saddle"
    else:
        # 无法确定
        return "Indeterminate"

# 分析和统计临界点的分类结果
def analyze_classification(classification):
    """
    分析和统计临界点的分类结果。

    参数:
    - classification (list): 临界点的分类结果列表

    返回:
    - None
    """
    focus_count = 0
    saddle_count = 0
    node_count = 0
    for classi in classification:
        for point_type in classi:
            if point_type == "Focus":
                focus_count += 1
            elif point_type == "Saddle":
                saddle_count += 1
            else:
                node_count += 1

    # print(f"Total singularity points: {len(classification)}")
    print(f"Focus: {focus_count}")
    print(f"Saddle: {saddle_count}")
    print(f"Node: {node_count}")


# 在每个时间步的速度场中找到临界点,并返回它们的坐标列表。
def find_singularity_points_for_all_Vk(V_k_coord, coordinates, triangles, eps):
    """
    在每个时间步的速度场中找到临界点,并返回它们的坐标列表。

    参数:
    - V_k_coord (list): k个时间步的速度场
    - coordinates (numpy.ndarray): 网格顶点坐标
    - triangles (numpy.ndarray): 网格三角形信息
    - eps (float): 临界点判断阈值

    返回:
    - list: 包含每个时间步的临界点坐标的列表
    """
    singularity_points = []
    for i, V_now in enumerate(V_k_coord):
        singularity_vertices, singularity_interiors, v_length_max = find_singularity_points(
            coordinates, triangles, V_now, eps)
        singularity_num = len(singularity_vertices) + len(
            singularity_interiors)
        print(f"第{i}个时刻临界点个数为{singularity_num}")

        singularity_points_now = []
        for singularity_vertice in singularity_vertices:
            singularity_points_now.append(singularity_vertice[1])
        for singularity_interior in singularity_interiors:
            singularity_points_now.append(singularity_interior[1])

        singularity_points.append(singularity_points_now)
    return singularity_points

# 在每个时间步的速度场中找到临界点,并对其进行分类。
def find_singularity_points_and_classify_for_all_Vk(V_k_coord, coordinates,
                                                    triangles, eps, surface,
                                                    e):
    """
    在每个时间步的速度场中找到临界点,并对其进行分类。

    参数:
    - V_k_coord (list): k个时间步的速度场
    - coordinates (numpy.ndarray): 网格顶点坐标
    - triangles (numpy.ndarray): 网格三角形信息
    - eps (float): 临界点判断阈值
    - surface (object): 三角形网格表面对象
    - e (numpy.ndarray): 每个顶点的正交基

    返回:
    - tuple:
        - singularity_points (list): 包含每个时间步的临界点坐标的列表
        - classification (list): 包含每个时间步的临界点分类结果的列表
    """
    singularity_points = []
    classification = []
    for i, V_now in enumerate(V_k_coord):
        singularity_vertices, singularity_interiors, v_length_max = find_singularity_points(
            coordinates, triangles, V_now, eps)
        singularity_num = len(singularity_vertices) + len(
            singularity_interiors)
        print(f"第{i}个时刻临界点个数为{singularity_num}")

        singularity_points_now = []
        classification_now = []
        for singularity_vertice in singularity_vertices:
            singularity_points_now.append(singularity_vertice[1])
            jacobian_matrix = compute_jacobian_matrix_for_vertex(
                singularity_vertice, V_now, surface, v_length_max, e)
            classification_now.append(classify_critical_point(jacobian_matrix))
        for singularity_interior in singularity_interiors:
            singularity_points_now.append(singularity_interior[1])
            jacobian_matrix = compute_jacobian_matrix_for_interior(
                singularity_interior, V_now, surface, v_length_max)
            classification_now.append(classify_critical_point(jacobian_matrix))

        singularity_points.append(singularity_points_now)
        classification.append(classification_now)

    return singularity_points, classification

# [用于模拟数据]计算检测到的临界点和实际临界点之间的测地距离
def compute_displacement_difference(threshold, singularity_points,
                                    true_singularity_points, surface, i, id):
    """
    计算检测到的临界点和实际临界点之间的测地距离,并返回相关统计信息。

    参数:
    - threshold (float): 真实临界点与检测到的临界点相对应的测地距离阈值
    - singularity_points (list): 检测到的临界点坐标列表
    - true_singularity_points (list): 实际的临界点坐标列表
    - surface (object): 三角形网格表面对象
    - i (int): 当前时间步
    - id (int): 目标临界点的时间步

    返回:
    - tuple: 包含以下元素的元组:
        - err (float): 测地距离差的总和
        - err_list (list): 每个匹配临界点的测地距离差列表
        - matched_num (int): 匹配到的临界点数量
        - spare_singularity_points_num (int): 多余的检测到的临界点数量
        - missed_singularity_points_num (int): 未检测到的临界点数量
    """
    err = 0
    err_list = []
    matched_num = 0
    spare_singularity_points_num = 0
    missed_singularity_points_num = 0

    true_singularity_num = len(true_singularity_points)
    singularity_num = len(singularity_points)

    true_singularity_vertices = [
        surface.find_closest_point(point) for point in true_singularity_points
    ]
    singularity_vertices = [
        surface.find_closest_point(point) for point in singularity_points
    ]
    flag = [False] * singularity_num

    if singularity_num == 0:
        missed_singularity_points_num = true_singularity_num
    else:
        if i < id:
            for true_vertex in true_singularity_vertices:
                displacement_difference = [
                    surface.geodesic_distance(true_vertex, vertex)
                    for vertex in singularity_vertices
                ]
                min_diff = min(displacement_difference)
                min_index = displacement_difference.index(min_diff)

                if min_diff <= threshold and flag[min_index] is False:
                    err_list.append(min_diff)
                    err += min_diff
                    flag[min_index] = True
                else:
                    missed_singularity_points_num += 1

            matched_num = flag.count(True) + 1
            spare_singularity_points_num = max(
                singularity_num - matched_num - 1, 0)
        else:
            missed_singularity_points_num = 2
            matched_num = 1

    return err, err_list, matched_num, spare_singularity_points_num, missed_singularity_points_num

# [用于模拟数据]计算检测到的临界点和实际临界点之间的总位移差异,以及其他相关统计指标。
def compute_err_for_all_Vk(true_singularity_points, singularity_points,
                           threshold, surface, turning_point):
    """
    计算检测到的临界点和实际临界点之间的总位移差异,以及其他相关统计指标。

    参数:
    - true_singularity_points (list): 每个时间步的实际临界点坐标列表
    - singularity_points (list): 每个时间步检测到的临界点坐标列表
    - threshold (float): 位移差异的阈值
    - surface (object): 三角形网格表面对象
    - turning_point (int): 目标临界点的时间步

    返回:
    - tuple:
        - err (float): 位移差异的总和
        - err_max (float): 最大位移差异
        - err_min (float): 最小位移差异
        - err_stdev (float): 位移差异的标准差
        - spare_singularity_points_num (int): 多余的检测到的临界点数量
        - missed_singularity_points_num (int): 未检测到的临界点数量
        - matched_num (int): 匹配到的临界点数量
    """
    err = 0
    err_list = []
    matched_num = 0
    spare_singularity_points_num = 0
    missed_singularity_points_num = 0

    for i, (singularity_points_now, true_singularity_points_now) in enumerate(
            zip(singularity_points, true_singularity_points)):
        # turning_point = 67
        err_now, err_list_now, matched_num_now, spare_singularity_points_num_now, missed_singularity_points_num_now = compute_displacement_difference(
            threshold, singularity_points_now, true_singularity_points_now,
            surface, i, turning_point)

        err += err_now
        err_list.extend(err_list_now)
        matched_num += matched_num_now
        spare_singularity_points_num += spare_singularity_points_num_now
        missed_singularity_points_num += missed_singularity_points_num_now

    err_max = max(err_list)
    err_min = min(err_list)
    err_stdev = statistics.stdev(err_list)

    return err, err_max, err_min, err_stdev, spare_singularity_points_num, missed_singularity_points_num, matched_num


if __name__ == "__main__":
    with open("./config/opticalflow.yaml", 'r', encoding='UTF-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data_params    = config['sub_08']
    general_params = config['general']

    e_path                  = data_params['e_path']
    V_k_path                = data_params['V_k_path']
    surface_path            = data_params['surface_path']
    potentials_path         = data_params['potentials_path']
    singularity_points_path = data_params['singularity_points_path']
    eps                     = general_params['eps']
    time_steps              = general_params['time_steps']
    # threshold                    = simulated_config['threshold']
    # true_singularity_points_path = simulated_config['true_singularity_points_path']


    e           = load_data(e_path).reshape(-1, 2, 3)
    V_k         = load_data(V_k_path)
    V_k_coord   = process_V_k(V_k, e)
    potentials  = load_data(potentials_path)
    surface     = pv.read(surface_path)
    coordinates = surface.points
    triangles   = surface.faces.reshape(-1, 4)[:, 1:]
    point_num   = len(coordinates)

    # with open(true_singularity_points_path, 'rb') as file:
    #     true_singularity_points = pickle.load(file)

    singularity_points = find_singularity_points_for_all_Vk(
        V_k_coord, coordinates, triangles, eps)

    # singularity_points, classification = find_singularity_points_and_classify_for_all_Vk(V_k_coord, coordinates, triangles, eps, surface)

    # print(classification)
    print(singularity_points)
    with open(singularity_points_path, 'wb') as file:
        pickle.dump(singularity_points, file)


    # analyze_classification(classification)


    ###################################### [用于模拟数据] ######################################
    # turning_point = 67
    # err, err_max, err_min, err_stdev, spare_singularity_points_num, missed_singularity_points_num, matched_num = compute_err_for_all_Vk(true_singularity_points, singularity_points, threshold, surface, turning_point)
    # print(f"总的err为{err}, 平均每个时间步长的err为{err / (time_steps - 1)}, 平均每个临界点的err为{err / matched_num}")
    # print(f"最大的err为{err_max}, 最小的err为{err_min}, err的标准差为{err_stdev}")
    # print(f"总的检测到的多余临界点数目为{spare_singularity_points_num}, 平均每个时间步长的多余临界点数目为{spare_singularity_points_num / (time_steps - 1)}")
    # print(f"总的未检测到的临界点数目为{missed_singularity_points_num}, 平均每个时间步长的未检测到的临界点数目为{missed_singularity_points_num / (time_steps - 1)}")
