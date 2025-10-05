import re
import yaml
import pickle
import pyvista as pv
import sys
import os
import json
import bz2
import numpy as np

from utils import compute_optical_flow, find_singularity_point

if "__main__" == __name__:
    data_path    = '/fred/oz284/mc/data/ds004080'
    results_path = '/fred/oz284/mc/results/ds004080'
    subfolder  = sys.argv[1]
    run_num    = sys.argv[2]
    trial_name = sys.argv[3]

    with open("config.yaml", 'r', encoding='UTF-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    general_params = config['general']
    # data_params    = config['sub_01']

    lambda_       = general_params['lambda_']  # 光流法正则化参数(速度场平滑项参数)0.01
    eps           = general_params['eps']    # 判断速度是否为0的误差
    # time_steps    = general_params['time_steps'] # 时间步数
    processes_num = general_params['processes_num']  # 进程池大小

    # surface_path             = data_params['surface_path']  # 三角剖分的表面文件路径
    # potentials_path          = data_params['potentials_path']  # 电势数据文件路径
    # e_path                   = data_params['e_path']  # 基底向量保存路径
    # V_k_path                 = data_params['V_k_path']  # 速度场保存路径
    # singularity_points_path  = data_params['singularity_points_path']  # 检测到的临界点保存路径
    # velocity_fields_gif_path = data_params['velocity_fields_gif_path']  # 光流场绘制保存路径


    data_subfolder_path = f"{data_path}/{subfolder}"
    filelist            = os.listdir(data_subfolder_path)
    ses                 = filelist[0]

    surface_path = rf'{results_path}/{subfolder}/{subfolder}_reconstructed_surface.ply'
    potentials_path = f"{results_path}/{subfolder}/run-{run_num}/{trial_name}/{subfolder}_{ses}_task-SPESclin_run-{run_num}-{trial_name}-ave-interpolation_data.csv"
    e_path     = f"{results_path}/{subfolder}/{subfolder}_e.csv"
    V_k_path = f"{results_path}/{subfolder}/run-{run_num}/{trial_name}/{subfolder}_{ses}_task-SPESclin_run-{run_num}-{trial_name}-V_k.csv"
    json_data = f"{data_path}/{subfolder}/{ses}/ieeg/{subfolder}_{ses}_task-SPESclin_run-{run_num}_ieeg.json"

    singularity_points_path = f"{results_path}/{subfolder}/run-{run_num}/{trial_name}/{subfolder}_{ses}_task-SPESclin_run-{run_num}-{trial_name}-singularity_points.pkl"
    classification_path = f'{results_path}/{subfolder}/run-{run_num}/{trial_name}/{subfolder}_{ses}_task-SPESclin_run-{run_num}-{trial_name}-singularity_points_classification.pkl'

    # 波速保存
    sl_fname = f"{results_path}/{subfolder}/run-{run_num}/{trial_name}/{subfolder}_{ses}_task-SPESclin_run-{run_num}-{trial_name}-wave_velocity_opticalflow.pkl.bz2"

    with open(json_data,'r',encoding='UTF-8') as f:
        json_info = json.load(f)
    SF = round(json_info["SamplingFrequency"])

    # e_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-e.csv"
    # V_k_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-V_k.csv"
    # singularity_points_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-singularity_points.pkl"
    # velocity_fields_gif_path = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-velocity_fields.gif"


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

    time_steps = len(potentials)
    t_k = [i / SF for i in range(time_steps)]
    t_k_ = [i for i in range(time_steps)]
    I_k = potentials[t_k_]
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
    V_k_coord = find_singularity_point.process_V_k(V_k, e)

    V_k_coord = np.array(V_k_coord)
    # 使用 NumPy 向量化操作来计算 V_c
    V_c = np.sqrt(np.sum(V_k_coord[:, :, :3] ** 2, axis=2))  # 计算每个坐标的模长

    
    # sl_fname = f"/fred/oz284/mc/results/CCEP-sub-01/{sys.argv[1]}/sub_01-{sys.argv[1]}-wave_velocity.pkl.bz2"
    with bz2.BZ2File(sl_fname, 'wb') as file:
        pickle.dump(V_c, file)

    # singularity_points = final_find_sigularity_point.find_singularity_points_for_all_Vk(
    #     V_k_coord, 
    #     coordinates, 
    #     triangles, 
    #     eps
    # )
    # singularity_points, classification = final_find_sigularity_point.find_singularity_points_and_classify_for_all_Vk(
    #     V_k_coord, 
    #     coordinates, 
    #     triangles, 
    #     eps, 
    #     surface, 
    #     e
    # )
    # with open(singularity_points_path, 'wb') as file:
    #     pickle.dump(singularity_points, file)
    # with poen(classification_path, 'wb') as file:
    #     pickle.dump(classification, file)


    ############################################### 绘制光流场 ###############################################
    # final_draw_optical_flow_field.plot_velocity_fields_and_singularity_points_gif(surface, potentials, V_k_coord, singularity_points, velocity_fields_gif_path)




    ################################# 模拟数据计算真实临界点和检测临界点的位移误差(测地距离) #################################
    # turning_point = 67
    # with open(true_singularity_points_path, 'rb') as file:
    #     true_singularity_points = pickle.load(file)
    # err, err_max, err_min, err_stdev, spare_singularity_points_num, missed_singularity_points_num, matched_num = compute_err_for_all_Vk(true_singularity_points, singularity_points, threshold, surface, turning_point)
    # print(f"总的err为{err}, 平均每个时间步长的err为{err / (time_steps - 1)}, 平均每个临界点的err为{err / matched_num}")
    # print(f"最大的err为{err_max}, 最小的err为{err_min}, err的标准差为{err_stdev}")
    # print(f"总的检测到的多余临界点数目为{spare_singularity_points_num}, 平均每个时间步长的多余临界点数目为{spare_singularity_points_num / (time_steps - 1)}")
    # print(f"总的未检测到的临界点数目为{missed_singularity_points_num}, 平均每个时间步长的未检测到的临界点数目为{missed_singularity_points_num / (time_steps - 1)}")
