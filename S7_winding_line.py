import math
import pickle
import pyvista as pv
import numpy as np
import yaml
from functools import partial
from utils import draw_optical_flow_field
import sys
import bz2

# 计算法向量n_i对应切平面的正交基底
def compute_orthonormal_basis(n_i):
    """
    计算法向量n_i对应切平面的正交基底(e1, e2)。
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
    """
    n = np.cross(e1, e2)
    Vn = np.dot(V, n) * n / np.dot(n, n)
    Vt = V - Vn
    return Vt

# 计算B点相对于A点, 在(e1, e2)方向上分别的位置差
def position_diff_on_basis_with_origin(A, B, e1, e2):
    """
    计算点B相对于点A在基底(e1, e2)定义的平面上的位置差。
    """
    B_relative = B - A
    n = np.cross(e1, e2)
    proj = B_relative - np.dot(B_relative, n) * n / np.dot(n, n)
    u = np.dot(proj, e1)
    v = np.dot(proj, e2)
    return u, v

# 把(e1, e2)平面上的向量V,用基底线性组合的方式表示
def express_vector_on_basis(V, e1, e2):
    """
    表示向量V在由e1和e2构成的基底上的线性组合。
    """
    V, e1, e2 = map(np.array, (V, e1, e2))
    if np.all(e1 == 0) or np.all(e2 == 0):
        raise ValueError("基底向量不能是零向量。")
    alpha = np.dot(V, e1) / np.dot(e1, e1)
    beta = np.dot(V, e2) / np.dot(e2, e2)
    return alpha, beta

def angle_between_vectors(v1, v2):
    """
    计算两个向量v1和v2之间的有序角度差(以弧度为单位)v2-v1，逆时针为正
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_u, v2_u)
    if dot_product > 1:
        dot_product = 1
    elif dot_product < -1:
        dot_product = -1
    angle = np.arccos(dot_product)
    cross_product = np.cross(v1_u, v2_u)
    if cross_product < 0:
        angle = -angle
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
    values = [math.atan2(y[i], x[i]) for i in range(n)]
    sorted_indices = np.lexsort((values, ))
    if vx is not None and vy is not None:
        return sorted_indices, x[sorted_indices], y[sorted_indices], vx[sorted_indices], vy[sorted_indices]
    else:
        return sorted_indices, x[sorted_indices], y[sorted_indices]

def check_property(data, flag):
    """
    判断绕圈数是否满足奇点性质
    flag=1为节点，flag=-1为鞍
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
    返回每个奇点的计数和类型
    """
    winding_numbers_counts = []
    types = []
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
    计算scale值映射函数，用于可视化
    """
    scale = np.zeros(len(points))
    all_points = []
    for idx, singularity in enumerate(singularity_points):
        index = surf.find_closest_point(singularity)
        scale[index] = winding_numbers_counts[idx] + 1
        point_neighbors_levels = surf.point_neighbors_levels(index, winding_numbers_counts[idx])
        point_neighbors_levels = list(point_neighbors_levels)
        for i in range(winding_numbers_counts[idx]):
            nei = point_neighbors_levels[i]
            for x in nei:
                scale[x] = winding_numbers_counts[idx] - i
                all_points.append(points[x])
    return scale, all_points

# 可视化函数（如需可视化可取消注释）
# def visualize_results(surf, winding_numbers_counts, types, all_points):
#     surf['scale'] = scale
#     max_scale = max(winding_numbers_counts)
#     contours = surf.contour(isosurfaces=np.arange(1.0, float(max_scale) + 1), scalars='scale')
#     p = pv.Plotter()
#     p.add_mesh(contours)
#     p.add_mesh(surf, show_edges=False)
#     # p.add_mesh(pv.PolyData(all_points), color='red', point_size=10)
#     p.show()

if "__main__" == __name__:
    # 读取配置文件
    with open("config.yaml", 'r', encoding='UTF-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    data_params = config['sub_01']
    general_params = config['general']
    surface_path = data_params['surface_path']

    # 路径参数
    potentials_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-ave-interpolation_data.csv"
    e_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-e.csv"
    V_k_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-V_k.csv"
    singularity_points_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-singularity_points.pkl"

    trial_name = sys.argv[1]

    # 加载数据
    e          = draw_optical_flow_field.load_data(e_path).reshape(-1, 2, 3)
    V_k        = draw_optical_flow_field.load_data(V_k_path)
    V_k_coord  = draw_optical_flow_field.process_V_k(V_k, e)
    potentials = draw_optical_flow_field.load_data(potentials_path)

    with open(singularity_points_path, 'rb') as file:
        singularity_points = pickle.load(file)

    surf    = pv.read(surface_path)
    normals = surf.point_normals
    points  = surf.points

    # 统计每个时刻的奇点及其绕圈数
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
    
    # 保存结果
    sl_fname = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-winding_lines.pkl.bz2"
    with bz2.BZ2File(sl_fname, 'wb') as file:
        pickle.dump(winding_lines, file)

    # # 单帧测试代码
    # time_index = 0
    # V_now = V_k_coord[time_index]
    # singularity_points = singularity_points[time_index]
    # print(f"singularity_points:{singularity_points}")
    # winding_numbers_counts, types = calculate_winding_numbers(surf, singularity_points, V_now, e, points)
    # print(winding_numbers_counts, types)
    # # scale, all_points = calculate_scale_values(surf, singularity_points, winding_numbers_counts, points)
    # # visualize_results(surf, winding_numbers_counts, types, all_points)