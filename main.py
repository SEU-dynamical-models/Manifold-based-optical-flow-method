import yaml
import pickle
import pyvista as pv

import compute_optical_flow
import find_sigularity_point
import draw_optical_flow_field

if "__main__" == __name__:
    with open("./config/config.yaml", 'r', encoding='UTF-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    general_params = config['general']
    data_params    = config['sub_01']

    lambda_       = general_params['lambda_']  # 光流法正则化参数(速度场平滑项参数)0.01
    eps           = general_params['eps']    # 判断速度是否为0的误差
    time_steps    = general_params['time_steps'] # 时间步数
    processes_num = general_params['processes_num']  # 进程池大小

    surface_path             = data_params['surface_path']  # 三角剖分的表面文件路径
    potentials_path          = data_params['potentials_path']  # 电势数据文件路径
    e_path                   = data_params['e_path']  # 基底向量保存路径
    V_k_path                 = data_params['V_k_path']  # 速度场保存路径
    processed_surface_path   = data_params['processed_surface_path']  # 几何相关数信息保存路径
    singularity_points_path  = data_params['singularity_points_path']  # 检测到的临界点保存路径
    velocity_fields_gif_path = data_params['velocity_fields_gif_path']  # 光流场绘制保存路径

    # surface_path                 = data_params['surface_path']
    # potentials_path              = data_params['potentials_path']
    # e_path                       = data_params['e_path']
    # V_k_path                     = data_params['V_k_path']
    # processed_surface_path       = data_params['processed_surface_path']
    # potentials_path_2            = data_params['potentials_path_2']
    # singularity_points_path      = data_params['singularity_points_path']
    # threshold                    = config['threshold']
    # true_singularity_points_path = config['true_singularity_points_path']

    surface    = pv.read(surface_path)
    potentials = compute_optical_flow.load_potentials(potentials_path)
    # potentials_2           = load_potentials(potentials_path_2)

    coordinates = surface.points
    triangles   = surface.faces.reshape(-1, 4)[:, 1:]
    point_num   = len(coordinates)
    mesh_num    = len(triangles)
    normals     = surface.point_normals
    areas       = surface.compute_cell_sizes(length=False, volume=False)['Area']

    t_k = [i for i in range(time_steps)]
    I_k = potentials[t_k]
    # I_k_2 = potentials_2[t_k]

    print("结点个数：", len(surface.points))
    print("三角形面片个数：", len(triangles))


    ############################################### 计算光流场 ###############################################
    a2, grad_w, e, integral_wi_wj, execution_time = compute_optical_flow.compute_geometrical_quantities(
        coordinates, 
        normals, 
        triangles, 
        areas
    )
    print("a2, grad_w, e, integral_wi_wj计算完成,花费时间为: ", execution_time)
    V_k, execution_time = compute_optical_flow.compute_velocity_field(
        processes_num, 
        time_steps, 
        a2, 
        grad_w, 
        e, 
        integral_wi_wj, 
        triangles,
        t_k, 
        areas, 
        lambda_, 
        I_k, 
        I_k
    )
    print("总的速度场计算时间：", execution_time, "秒")
    print("平均每个速度场计算时间：", execution_time / (time_steps - 1), "秒")

    # 保存数据到CSV文件中
    compute_optical_flow.reshape_and_save_data(e, e_path)  # 保存正交基底e
    compute_optical_flow.reshape_and_save_data(V_k, V_k_path)  # 保存速度场V_k



    ############################################### 检测临界点 ###############################################
    V_k_coord = find_sigularity_point.process_V_k(V_k, e)
    singularity_points = find_sigularity_point.find_singularity_points_for_all_Vk(
        V_k_coord, 
        coordinates, 
        triangles, 
        eps
    )
    # singularity_points, classification = find_sigularity_point.find_singularity_points_and_classify_for_all_Vk(
    #     V_k_coord, 
    #     coordinates, 
    #     triangles, 
    #     eps, 
    #     surface, 
    #     e
    # )
    with open(singularity_points_path, 'wb') as file:
        pickle.dump(singularity_points, file)



    ############################################### 绘制光流场 ###############################################
    draw_optical_flow_field.plot_velocity_fields_and_singularity_points_gif(surface, potentials, V_k_coord, singularity_points, velocity_fields_gif_path)




    ################################# 模拟数据计算真实临界点和检测临界点的位移误差(测地距离) #################################
    # turning_point = 67
    # with open(true_singularity_points_path, 'rb') as file:
    #     true_singularity_points = pickle.load(file)
    # err, err_max, err_min, err_stdev, spare_singularity_points_num, missed_singularity_points_num, matched_num = compute_err_for_all_Vk(true_singularity_points, singularity_points, threshold, surface, turning_point)
    # print(f"总的err为{err}, 平均每个时间步长的err为{err / (time_steps - 1)}, 平均每个临界点的err为{err / matched_num}")
    # print(f"最大的err为{err_max}, 最小的err为{err_min}, err的标准差为{err_stdev}")
    # print(f"总的检测到的多余临界点数目为{spare_singularity_points_num}, 平均每个时间步长的多余临界点数目为{spare_singularity_points_num / (time_steps - 1)}")
    # print(f"总的未检测到的临界点数目为{missed_singularity_points_num}, 平均每个时间步长的未检测到的临界点数目为{missed_singularity_points_num / (time_steps - 1)}")
