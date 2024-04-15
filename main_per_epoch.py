import os
import mne
import yaml
import pickle
import pyvista as pv

import compute_optical_flow
import find_sigularity_point
import draw_optical_flow_field


trials = ["VERTICAL-L-R-1", "VERTICAL-L-R-2", "VERTICAL-L-R-3", "VERTICAL-L-R-4", "VERTICAL-L-R-5", "VERTICAL-L-R-6", "VERTICAL-L-R-7", "VERTICAL-L-R-8", "VERTICAL-L-R-9", "VERTICAL-L-R-10", "VERTICAL-L-R-11", "VERTICAL-L-R-12", "VERTICAL-L-R-13", "VERTICAL-L-R-14", "VERTICAL-L-R-15", "VERTICAL-L-R-16", "VERTICAL-L-R-17", "VERTICAL-L-R-18", "VERTICAL-L-R-19", "VERTICAL-L-R-20", "VERTICAL-L-R-21", "VERTICAL-L-R-22", "VERTICAL-L-R-23", "VERTICAL-L-R-24", "VERTICAL-L-R-25", "VERTICAL-L-R-26", "VERTICAL-L-R-27", "VERTICAL-L-R-28", "DIAGONAL-RD-LU-1", "DIAGONAL-RD-LU-2", "DIAGONAL-RD-LU-3", "DIAGONAL-RD-LU-4", "DIAGONAL-RD-LU-5", "DIAGONAL-RD-LU-6", "DIAGONAL-RD-LU-7", "DIAGONAL-RD-LU-8", "DIAGONAL-RD-LU-9", "DIAGONAL-RD-LU-10", "DIAGONAL-RD-LU-11", "DIAGONAL-RD-LU-12", "BLANK", "HORIZONTAL-U-D-1", "HORIZONTAL-U-D-2", "HORIZONTAL-U-D-3", "HORIZONTAL-U-D-4", "HORIZONTAL-U-D-5", "HORIZONTAL-U-D-6", "HORIZONTAL-U-D-7", "HORIZONTAL-U-D-8", "HORIZONTAL-U-D-9", "HORIZONTAL-U-D-10", "HORIZONTAL-U-D-11", "HORIZONTAL-U-D-12", "HORIZONTAL-U-D-13", "HORIZONTAL-U-D-14", "HORIZONTAL-U-D-15", "HORIZONTAL-U-D-16", "HORIZONTAL-U-D-17", "HORIZONTAL-U-D-18", "HORIZONTAL-U-D-19", "HORIZONTAL-U-D-20", "HORIZONTAL-U-D-21", "HORIZONTAL-U-D-22", "HORIZONTAL-U-D-23", "HORIZONTAL-U-D-24", "HORIZONTAL-U-D-25", "HORIZONTAL-U-D-26", "HORIZONTAL-U-D-27", "HORIZONTAL-U-D-28", "DIAGONAL-LD-RU-1", "DIAGONAL-LD-RU-2", "DIAGONAL-LD-RU-3", "DIAGONAL-LD-RU-4", "DIAGONAL-LD-RU-5", "DIAGONAL-LD-RU-6", "DIAGONAL-LD-RU-7", "DIAGONAL-LD-RU-8", "DIAGONAL-LD-RU-9", "DIAGONAL-LD-RU-10", "DIAGONAL-LD-RU-11", "DIAGONAL-LD-RU-12", "DIAGONAL-LU-RD-1", "DIAGONAL-LU-RD-2", "DIAGONAL-LU-RD-3", "DIAGONAL-LU-RD-4", "DIAGONAL-LU-RD-5", "DIAGONAL-LU-RD-6", "DIAGONAL-LU-RD-7", "DIAGONAL-LU-RD-8", "DIAGONAL-LU-RD-9", "DIAGONAL-LU-RD-10", "DIAGONAL-LU-RD-11", "DIAGONAL-LU-RD-12", "DIAGONAL-RU-LD-1", "DIAGONAL-RU-LD-2", "DIAGONAL-RU-LD-3", "DIAGONAL-RU-LD-4", "DIAGONAL-RU-LD-5", "DIAGONAL-RU-LD-6", "DIAGONAL-RU-LD-7", "DIAGONAL-RU-LD-8", "DIAGONAL-RU-LD-9", "DIAGONAL-RU-LD-10", "DIAGONAL-RU-LD-11", "DIAGONAL-RU-LD-12"]

if "__main__" == __name__:
    with open("./config/config.yaml", 'r', encoding='UTF-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    general_params = config['general']
    # data_params    = config['sub_01']

    lambda_       = general_params['lambda_']  # 光流法正则化参数(速度场平滑项参数)0.01
    eps           = general_params['eps']    # 判断速度是否为0的误差
    time_steps    = general_params['time_steps'] # 时间步数
    processes_num = general_params['processes_num']  # 进程池大小

    # surface_path             = data_params['surface_path']  # 三角剖分的表面文件路径
    # potentials_path          = data_params['potentials_path']  # 电势数据文件路径
    # e_path                   = data_params['e_path']  # 基底向量保存路径
    # V_k_path                 = data_params['V_k_path']  # 速度场保存路径
    # processed_surface_path   = data_params['processed_surface_path']  # 几何相关数信息保存路径
    # singularity_points_path  = data_params['singularity_points_path']  # 检测到的临界点保存路径
    # velocity_fields_gif_path = data_params['velocity_fields_gif_path']  # 光流场绘制保存路径

    # surface_path                 = data_params['surface_path']
    # potentials_path              = data_params['potentials_path']
    # e_path                       = data_params['e_path']
    # V_k_path                     = data_params['V_k_path']
    # processed_surface_path       = data_params['processed_surface_path']
    # potentials_path_2            = data_params['potentials_path_2']
    # singularity_points_path      = data_params['singularity_points_path']
    # threshold                    = config['threshold']
    # true_singularity_points_path = config['true_singularity_points_path']


    surface_path = r"results\sub-p08\preprocessed_data\sub-p08_reconstructed_surface.ply"
    epochs_path = r"results\sub-p08\preprocessed_data\sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01-epo.fif"

    surface = pv.read(surface_path)
    epochs  = mne.read_epochs(epochs_path, preload=False)

    coordinates = surface.points
    triangles   = surface.faces.reshape(-1, 4)[:, 1:]
    point_num   = len(coordinates)
    mesh_num    = len(triangles)
    normals     = surface.point_normals
    areas       = surface.compute_cell_sizes(length=False, volume=False)['Area']

    print("结点个数：", len(surface.points))
    print("三角形面片个数：", len(triangles))

    for trial in trials:
        folder_path = "results/sub-p08/optical_flow/sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_" + trial
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"{folder_path} created")
        epochs_trial = epochs[trial]
        for i, potentials in enumerate(epochs_trial):
            e_path = folder_path + "/trial_0{}_e.csv".format(i+1)
            V_k_path = folder_path + "/trial_0{}_V_k.csv".format(i+1)
            singularity_points_path = folder_path + "/trial_0{}_singularity_points.pkl".format(i+1)
            singularity_points_classification_path = folder_path + "/trial_0{}_singularity_points_classification.pkl".format(i+1)
            velocity_fields_gif_path = folder_path + "/trial_0{}_velocity_fields.gif".format(i+1)
            potentials_path = "results/sub-p08/preprocessed_data/trials_potentials/sub-p08_ses-nyuecog01_task-prf_acq-clinical_run-01_" + trial + "_0{}_potentials.csv".format(i+1)


            potentials = compute_optical_flow.load_potentials(potentials_path)
            # potentials_2           = load_potentials(potentials_path_2)

            t_k = [i for i in range(time_steps)]
            I_k = potentials[t_k]
            # I_k_2 = potentials_2[t_k]


            ############################################### 计算光流场 ###############################################
            a2, grad_w, e, integral_wi_wj, execution_time = compute_optical_flow.compute_geometrical_quantities(
                coordinates, 
                normals, 
                triangles, 
                areas
            )
            print(f"{trial}_0{i+1}: a2, grad_w, e, integral_wi_wj计算完成,花费时间为: ", execution_time)

            # 保存数据到CSV文件中
            compute_optical_flow.reshape_and_save_data(e, e_path)  # 保存正交基底e

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
            print(f"{trial}_0{i+1}: 总的速度场计算时间：", execution_time, "秒")
            print(f"{trial}_0{i+1}: 平均每个速度场计算时间：", execution_time / (time_steps - 1), "秒")

            
            # 保存数据到CSV文件中
            compute_optical_flow.reshape_and_save_data(V_k, V_k_path)  # 保存速度场V_k


            ############################################### 检测临界点 ###############################################
            V_k_coord = find_sigularity_point.process_V_k(V_k, e)
            # singularity_points = find_sigularity_point.find_singularity_points_for_all_Vk(
            #     V_k_coord, 
            #     coordinates, 
            #     triangles, 
            #     eps
            # )
            singularity_points, classification = find_sigularity_point.find_singularity_points_and_classify_for_all_Vk(
                V_k_coord, 
                coordinates, 
                triangles, 
                eps, 
                surface, 
                e
            )
            with open(singularity_points_path, 'wb') as file:
                pickle.dump(singularity_points, file)
            with open(singularity_points_classification_path, 'wb') as file:
                pickle.dump(classification, file)

            find_sigularity_point.analyze_classification(classification)

            ############################################### 绘制光流场 ###############################################
            # draw_optical_flow_field.plot_velocity_fields_and_singularity_points_gif(surface, potentials, V_k_coord, singularity_points, velocity_fields_gif_path)




            ################################# 模拟数据计算真实临界点和检测临界点的位移误差(测地距离) #################################
            # turning_point = 67
            # with open(true_singularity_points_path, 'rb') as file:
            #     true_singularity_points = pickle.load(file)
            # err, err_max, err_min, err_stdev, spare_singularity_points_num, missed_singularity_points_num, matched_num = compute_err_for_all_Vk(true_singularity_points, singularity_points, threshold, surface, turning_point)
            # print(f"总的err为{err}, 平均每个时间步长的err为{err / (time_steps - 1)}, 平均每个临界点的err为{err / matched_num}")
            # print(f"最大的err为{err_max}, 最小的err为{err_min}, err的标准差为{err_stdev}")
            # print(f"总的检测到的多余临界点数目为{spare_singularity_points_num}, 平均每个时间步长的多余临界点数目为{spare_singularity_points_num / (time_steps - 1)}")
            # print(f"总的未检测到的临界点数目为{missed_singularity_points_num}, 平均每个时间步长的未检测到的临界点数目为{missed_singularity_points_num / (time_steps - 1)}")
