import pickle
import numpy as np
import yaml
import draw_optical_flow_field
import pyvista as pv
import bz2
# import hdf5storage as hdf



def track_static_vectorfields_over_time(surface, V_k_coord, start_time_idx=50, end_time_idx=60, min_streamline_length=20):
    vectorfields_streamlines = {}  # 存储不同时间点的流线
    points = surface.points

    for time_idx in range(start_time_idx, end_time_idx):
        current_time_streamlines = []  # 存储当前时间点的流线
        V_now = V_k_coord[time_idx]  # 当前时间点的向量场

        for point_idx, point in enumerate(points):
            if np.linalg.norm(V_now[point_idx]) != 0:  # 检查向量场的长度是否为零
                streamline = extract_static_streamline_dot_product(point, V_now, surface, min_streamline_length)
                if len(streamline) != 0:
                    current_time_streamlines.append(streamline)

        vectorfields_streamlines[str(time_idx)] = current_time_streamlines

    return vectorfields_streamlines

def is_valid_direction(dots_idx, idx, point_neighbors, streamline):
    if dots_idx > 0:
        if any(np.array_equal(point_neighbors[idx], p) for p in streamline):
            return False
        else:
            return True
    else:
        return False


def extract_static_streamline_dot_product(point, V_now, surface, min_streamline_length=20):
    coordinates = surface.points  # 曲面顶点坐标
    normals     = surface.point_normals  # 曲面顶点法向量
    triangles   = surface.faces.reshape(-1, 4)[:, 1:]  # 曲面三角形单元

    next_point = surface.find_closest_point(point)  # 找到距离给定坐标最近的曲面顶点作为起点
    streamline = [coordinates[next_point]]  # 初始化流线点集合
    # print("streamline:", streamline)

    while True:
        n_i = normals[next_point]  # 当前顶点的法向量
        e1, e2 = compute_orthonormal_basis(n_i)  # 计算正交基

        cell_neighbors_id = surface.point_cell_ids(next_point)  # 当前顶点所在的三角单元索引
        point_neighbors_id = surface.point_neighbors(next_point)  # 当前顶点的相邻顶点索引
        point_neighbors = coordinates[point_neighbors_id]  # 当前顶点的相邻顶点坐标
        vectors = point_neighbors - coordinates[next_point]  # 当前顶点坐标到相邻顶点坐标的向量
        vectors_projection = [project_vector_to_plane(v, e1, e2) for v in vectors]  # 将向量投影到平面上
        normalized_vectors_projection = []  # 归一化投影向量
        for v in vectors_projection:
            length = np.linalg.norm(v) # 计算向量长度
            normalized_vector = v / length # 创建单位向量
            normalized_vectors_projection.append(normalized_vector)
        dots = np.sum(normalized_vectors_projection * V_now[next_point], axis=1)  # 计算点乘
        idx = np.argmax(dots)  # 找到最大点乘的索引


        if len(cell_neighbors_id) >= 6: # 位于曲面内部
            if is_valid_direction(dots[idx], idx, point_neighbors, streamline):
                next_point = point_neighbors_id[idx]
                streamline.append(point_neighbors[idx])
            else:
                break

        else: # 位于曲面边界
            cell_neighbors = triangles[cell_neighbors_id]
            choice_point = point_neighbors_id[idx]
            choice_point_cell_neighbors_id = surface.point_cell_ids(choice_point)  # 选择顶点所在的三角单元索引
            common_cell_neighbors_id = list((set(cell_neighbors_id) & set(choice_point_cell_neighbors_id)))  # 获取共同的单元格邻居

            # 共享多个单元格邻居
            if len(common_cell_neighbors_id) >= 2:  
                if is_valid_direction(dots[idx], idx, point_neighbors, streamline):  # 检查是否是有效的方向
                    next_point = point_neighbors_id[idx]  # 更新下一个顶点
                    streamline.append(point_neighbors[idx])  # 添加到流线点集合
                else:
                    break
            # 只有一个单元格邻居
            else: 
                coord = triangles[common_cell_neighbors_id]  # 共享的单元格邻居的三角形顶点坐标
                # print(coord)
                # 获取共享三角形的顶点
                A, B, C = coord[0]
                if A == next_point:
                    A = A
                elif B == next_point:
                    B = A
                    A = next_point
                elif C == next_point:
                    C = A
                    A = next_point
                else:
                    print("Error!")
                # 计算在基底上的位置差
                posx1, posy1 = position_diff_on_basis_with_origin(A, B, e1, e2)
                posx2, posy2 = position_diff_on_basis_with_origin(C, A, e1, e2)
                tri = [[0, 0], [posx1, posy1], [posx2, posy2]]
                # print(f"三角形为:{tri}")

                a, b, c = tri
                coord_order = clockwise(a, b, c)  # 逆时针排序
                if len(coord_order) == 0:
                    break
                else:
                    a, b, c = coord_order

                V_on_basis_x, V_on_basis_y = express_vector_on_basis(V_now[next_point], e1, e2)  # 将向量表示为基底分量
                # print(a, b, c, [V_on_basis_x, V_on_basis_y])

                ans = inTri(a, b, c, [V_on_basis_x, V_on_basis_y])  # 判断向量是否在三角形内部
                if ans == 1 and is_valid_direction(dots[idx], idx,point_neighbors, streamline):
                    next_point = point_neighbors_id[idx]  # 更新下一个顶点
                    streamline.append(point_neighbors[idx])  # 添加到流线点集合
                else:
                    break
                
    if len(streamline) >= min_streamline_length:
        return streamline
    else:
        return []


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


# 判断点p与三角形abc的位置关系
def inTri(a, b, c, p):
    # 将输入参数转换为NumPy数组
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    p = np.array(p)

    # 计算向量ab、ap、bc、bp、ca、cp
    ab = b - a
    ap = p - a
    bc = c - b
    bp = p - b
    ca = a - c
    cp = p - c

    # 判断点p是否在三角形abc内部、边上或外部
    if np.cross(ab, ap) > 0 and np.cross(bc, bp) > 0 and np.cross(ca, cp) > 0:
        # 在三角形内部，返回1
        return 1
    if np.cross(ab, ap) * np.cross(bc, bp) * np.cross(ca, cp) == 0:
        # 在三角形的边上，返回0
        return 0
    # 在三角形外部，返回-1
    return -1

# 返回三角形ABC的逆时针序列
def clockwise(A, B, C):
    x1, y1 = A
    x2, y2 = B 
    x3, y3 = C

    AB = (x2 - x1, y2 - y1)
    AC = (x3 - x1, y3 - y1)
    
    # 计算叉积
    cross = AB[0] * AC[1] - AB[1] * AC[0]
    
    if cross > 0:  # ABC为逆时针方向
        return [A, B, C]  
    elif cross < 0:  # ABC为顺时针方向
        return [A, C, B]  
    else:
        return []  # 共线

# 根据points构造并返回PolyData对象(流线)
def lines_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

if "__main__" == __name__:

    with open("./config/config.yaml", 'r', encoding='UTF-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    data_params = config['sub_08']

    surface_path                           = data_params['surface_path']
    potentials_path                        = data_params['potentials_path']
    e_path                                 = data_params['e_path']
    V_k_path                               = data_params['V_k_path']
    singularity_points_path                = data_params['singularity_points_path']
    singularity_points_classification_path = data_params['singularity_points_classification_path']


    e          = draw_optical_flow_field.load_data(e_path).reshape(-1, 2, 3)
    V_k        = draw_optical_flow_field.load_data(V_k_path)
    V_k_coord  = draw_optical_flow_field.process_V_k(V_k, e)
    potentials = draw_optical_flow_field.load_data(potentials_path)

    surface = pv.read(surface_path)


    # with open(singularity_points_path, 'rb') as file:
    #     singularity_points = pickle.load(file)
    # with open(singularity_points_classification_path, 'rb') as file:
    #     singularity_points_classification = pickle.load(file)
    # index = 65
    # singularity_points = singularity_points[index]
    # singularity_points_classification = singularity_points_classification[index]
    # print(f"singularity_points:{singularity_points}")
    # print(f"singularity_points_classification:{singularity_points_classification}")
    

    lines = []
    points = surface.points

    vectorfields_streamlines = track_static_vectorfields_over_time(surface, V_k_coord, start_time_idx=65, end_time_idx=66, min_streamline_length=15)

    sl_fname = "velocityfields_streamlines.pkl.bz2"
    with bz2.BZ2File(sl_fname, 'wb') as file:
        pickle.dump(vectorfields_streamlines, file)


    ############################# Visualization #############################
    sl_fname = "velocityfields_streamlines.pkl.bz2"
    with bz2.BZ2File(sl_fname, 'rb') as file:
        vectorfields_streamlines = pickle.load(file)

    
    for streamlines in vectorfields_streamlines.values():
        p = pv.Plotter()
        for streamline in streamlines:
            if len(streamline) != 0:
                line = pv.Spline(streamline, 100)
                line["scalars"] = np.arange(line.n_points)
                lines.append(line)
                p.add_mesh(line.tube(radius=0.1), cmap="bwr")
        p.show()

    ############################# Test extract_static_streamline_dot_product #############################
    # lines = []
    # points = surface.points
    # for point in points:
    #     streamline = extract_static_streamline_dot_product(point, V_k_coord[index], surface, 15)
    #     # print(streamline)

    #     # line = lines_from_points(streamline)
    #     # line["scalars"] = np.arange(line.n_points)
    #     # tube = line.tube(radius=0.1)
    #     # tube.plot(smooth_shading=True)
    #     if len(streamline) != 0:
    #         line = pv.Spline(streamline, 100)
    #         line["scalars"] = np.arange(line.n_points)

    #         lines.append(line)


    # p = pv.Plotter()
    # # p.add_mesh(surface, show_edges=False, smooth_shading=False)
    # for line in lines:
    #     p.add_mesh(line.tube(radius=0.1), cmap="Blues")
    # p.show()
