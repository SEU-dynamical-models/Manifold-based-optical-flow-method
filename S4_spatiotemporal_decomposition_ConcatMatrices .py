from matplotlib import pyplot as plt
import mne
import numpy as np
import draw_optical_flow_field
import pyvista as pv

def process_V_k_to_complex(V_k):
    """
    将V_k从实部和虚部拼接的形式还原为复数形式。
    V_k: shape (帧数, 2*点数)，前一半为实部，后一半为虚部。
    返回 shape (帧数, 点数) 的复数数组。
    """
    point_num = len(V_k[0]) // 2
    V_k_array = np.zeros((len(V_k), point_num, 2))
    for k, V_index in enumerate(V_k):
        for i in range(point_num):
            V_k_array[k][i] = [V_index[i], V_index[i + point_num]]
    V_k_coord_complex = np.empty((len(V_k_array), point_num), dtype=complex)
    for k, V_index in enumerate(V_k_array):
        for i, v in enumerate(V_index):
            V_k_coord_complex[k][i] = complex(v[0], v[1])
    return V_k_coord_complex

def calculate_V_k_from_complex(V_k_coord_complex, e):
    """
    将复数模态向量投影到空间坐标系上，得到每个点的速度矢量。
    V_k_coord_complex: shape (帧数, 点数) 的复数数组
    e: shape (帧数, 2, 3) 的正交基
    返回 shape (帧数, 3) 的速度矢量
    """
    V_k_coord = []
    for i, v in enumerate(V_k_coord_complex):
        V_1 = v.real * e[i][0]
        V_2 = v.imag * e[i][1]
        V_k_coord.append(V_1 + V_2)
    return V_k_coord

def plot_surface_with_velocity_arrows(surface_path, velocity, type='Raw', id=0, per=0):
    """
    绘制表面及其速度场箭头。
    type: 'Raw'为原始长度，'Scaled'为统一长度
    id/per: 模态编号和方差百分比
    """
    surface = pv.read(surface_path)
    surface['V'] = velocity
    if type == 'Raw':
        lengths = np.linalg.norm(velocity, axis=1)
        max_value = np.max(lengths)
        scale_factor = 10 / max_value
        scaled_lengths = lengths * scale_factor
        surface['V_scale'] = scaled_lengths
        args = {'tolerance': 0.01, 'factor': 0.2, 'scale': 'V_scale', 'orient': "V"}
    elif type == 'Scaled':
        surface['V_scale'] = np.ones((len(velocity), )) * 5
        args = {'tolerance': 0.01, 'factor': 0.7, 'scale': 'V_scale', 'orient': "V"}
    else:
        print("Wrong Type!")
        return
    p = pv.Plotter()
    arrows = surface.glyph(**args)
    p.add_mesh(arrows, color="black")
    p.add_mesh(surface, opacity=0.4, show_edges=False, smooth_shading=False, color=(227/255, 237/255, 249/255))
    p.add_text(f"Mode {id}, Var = {per}%", font_size=15)
    plot_pial_surfaces(r"ds004080\lh.pial", r"ds004080\rh.pial", p)
    p.view_yz(negative=True)
    p.show()

def calculate_percentages(Sigma):
    """
    计算每个奇异值的方差贡献率。
    返回两个百分比数组。
    """
    squared_Sigma = np.square(Sigma)
    sum_of_squared_Sigma = np.sum(squared_Sigma)
    sum_of_Sigma = np.sum(Sigma)
    percentages_squared = (squared_Sigma / sum_of_squared_Sigma) * 100
    percentages = (Sigma / sum_of_Sigma) * 100
    return np.round(percentages, 2), np.round(percentages_squared, 2)

def extract_modes(Sigma, VT, e, k, rounded_percentages_squared):
    """
    提取前k个模态并绘制其空间分布。
    """
    for t in range(k):
        sigma = Sigma[t]
        vt = VT[t, :]
        V_k_decomposition = calculate_V_k_from_complex(sigma * vt, e)
        plot_surface_with_velocity_arrows(surface_path, V_k_decomposition, "Scaled", t + 1, rounded_percentages_squared[t])

def plot_temporal_modes(U, nmodeplot, realTime):
    """
    绘制前nmodeplot个模态的时间变化曲线。
    """
    nt = len(realTime)
    re_uav = np.zeros((nt, nmodeplot))
    for it in range(nt):
        uav = U[it, :nmodeplot]
        re_uav[it, :] = np.real(uav)
    y_range = np.max(re_uav.ravel()) - np.min(re_uav.ravel())
    y_margin = 0.15 * y_range
    uav_lims = [np.min(re_uav.ravel()) - y_margin, np.max(re_uav.ravel()) + y_margin]
    colors = [
        (0/255, 114/255, 178/255),
        (0/255, 158/255, 115/255),
        (213/255, 94/255, 0/255),
        (230/255, 159/255, 0/255)
    ]
    plt.figure(figsize=(10, 8), dpi=100)
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })
    for imode in range(nmodeplot):
        plt.plot(
            realTime,
            re_uav[:, imode],
            color=colors[imode],
            label=f"Mode {imode + 1}",
            linewidth=3.0,
            solid_capstyle='round'
        )
    plt.xlabel('Time (s)', fontweight='bold')
    plt.ylabel('Temporal Mode', fontweight='bold')
    plt.tick_params(direction='out', length=6, width=1.5)
    plt.grid(False)
    plt.axvline(0, color='black', linewidth=1.5, alpha=0.6)
    plt.axhline(0, color='black', linewidth=1.5, alpha=0.6)
    plt.ylim(uav_lims)
    plt.xlim([realTime[0], realTime[-1]])
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right', frameon=True)
    plt.title('Temporal Evolution of SVD Modes', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('temporal_modes.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pial_surfaces(lh_pial_path, rh_pial_path, p):
    """
    绘制左右大脑皮层表面（pial surface）。
    """
    lh_verts, lh_faces = mne.read_surface(lh_pial_path)
    rh_verts, rh_faces = mne.read_surface(rh_pial_path)
    lh_faces = np.insert(lh_faces, 0, 3, axis=1)
    rh_faces = np.insert(rh_faces, 0, 3, axis=1)
    mesh1 = pv.PolyData(lh_verts, lh_faces)
    mesh2 = pv.PolyData(rh_verts, rh_faces)
    p.add_mesh(mesh1, color='grey', opacity=0.1, label='lh.pial', show_edges=False, smooth_shading=True, style='surface')
    # p.add_mesh(mesh2, color='grey', opacity=0.1, label='rh.pial', show_edges=False, smooth_shading=True, style='surface')

if "__main__" == __name__:
    # 参数设置
    trial_name = "PT47-PT48"
    surface_path = rf"ds004080\sub-ccepAgeUMCU01\sub-ccepAgeUMCU01_reconstructed_surface.ply"
    e_path = r"ds004080\sub-ccepAgeUMCU01\sub-ccepAgeUMCU01_e.csv"
    V_k_path = rf"ds004080\sub-ccepAgeUMCU01\run-021448\{trial_name}\sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448-{trial_name}-V_k.csv"
    save_npz = 'data_01.npz'
    nmodeplot = 4 # 选择前n个模态进行绘图

    # 加载正交基和速度场数据
    e = draw_optical_flow_field.load_data(e_path).reshape(-1, 2, 3)
    V_k = draw_optical_flow_field.load_data(V_k_path)
    V_k_coord = process_V_k_to_complex(V_k)

    # 拼接实部和虚部，用于SVD
    complex_array = V_k_coord
    point_num = len(V_k_coord[0])
    real_part = np.real(complex_array)
    imaginary_part = np.imag(complex_array)
    result_array = np.concatenate((real_part, imaginary_part), axis=1)

    # 奇异值分解
    U, Sigma, VT = np.linalg.svd(result_array, full_matrices=1)
    np.savez(save_npz, U=U, Sigma=Sigma, VT=VT, point_num=point_num)

    # 计算方差贡献率
    rounded_percentages, rounded_percentages_squared = calculate_percentages(Sigma)
    U = U[:, :nmodeplot]
    Sigma = Sigma[:nmodeplot]
    VT = VT[:nmodeplot, :]
    isNegative = np.mean(np.real(U), axis=0) < 0
    U[:, isNegative] = -U[:, isNegative]
    VT[isNegative, :] = -VT[isNegative, :]

    # 还原复数模态向量用于绘制
    V1 = VT[:, :point_num]
    V2 = VT[:, point_num:]
    V_final = V1 + V2 * 1j
    extract_modes(Sigma, V_final, e, nmodeplot, rounded_percentages_squared)

    # 绘制时间模态
    nt = len(V_k_coord)
    plot_temporal_modes(U, nmodeplot, np.linspace(0.009, 0.2, len(V_k_coord)))