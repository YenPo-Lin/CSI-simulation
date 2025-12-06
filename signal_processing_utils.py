import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from scipy.ndimage import gaussian_filter, maximum_filter

pic_width = 6
pic_height = 5
label_size = 10



class find_Peaks:

    def find_peak_aoa_doppler_interp(P_mvdr, theta, freqs, top_k=4, threshold_ratio=0.3):
        """
        找出 MVDR 頻譜矩陣中的主要峰值 (AoA, Doppler)，
        並利用二次拋物線內插法求 sub-bin 精確角度與頻率。

        Returns
        -------
        peaks_info : list of dict
            每個峰包含 {'theta': ..., 'freq': ..., 'value': ..., 
                    'theta_idx': i, 'freq_idx': j, 
                    'delta_theta': δ_i, 'delta_freq': δ_j}
        """
        P_abs = np.abs(P_mvdr)
        max_val = np.max(P_abs)
        mask = P_abs >= threshold_ratio * max_val

        peaks = []
        for i in range(1, P_abs.shape[0] - 1):
            for j in range(1, P_abs.shape[1] - 1):

                if mask[i, j] and P_abs[i, j] == np.max(P_abs[i-1:i+2, j-1:j+2]):
                    peaks.append((i, j, P_abs[i, j]))

        peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:top_k]
        peaks_info = []

        for (i, j, val) in peaks:
            # --- 1D quadratic interpolation for AoA axis (row) ---
            if 0 < i < P_abs.shape[0] - 1:
                y1, y2, y3 = P_abs[i - 1, j], P_abs[i, j], P_abs[i + 1, j]
                denom = 2 * (y1 - 2 * y2 + y3)
                delta_i = 0 if denom == 0 else (y1 - y3) / denom
            else:
                delta_i = 0

            # --- 1D quadratic interpolation for Doppler axis (col) ---
            if 0 < j < P_abs.shape[1] - 1:
                x1, x2, x3 = P_abs[i, j - 1], P_abs[i, j], P_abs[i, j + 1]
                denom = 2 * (x1 - 2 * x2 + x3)
                delta_j = 0 if denom == 0 else (x1 - x3) / denom
            else:
                delta_j = 0

            # --- convert to real-world angle/frequency ---
            theta_refined = theta[i] + delta_i * (theta[1] - theta[0])
            freq_refined = freqs[j] + delta_j * (freqs[1] - freqs[0])

            peaks_info.append({
                'theta': theta_refined,
                'freq': freq_refined,
                'value': val,
                'theta_idx': i,
                'freq_idx': j,
                'delta_theta': delta_i,
                'delta_freq': delta_j
            })

        return peaks_info

    def find_peak_aoa_doppler(P_mvdr, theta, freqs, top_k=2, threshold_ratio=0.05):   
        """
        找出 MVDR 頻譜矩陣中的主要峰值 (AoA, Doppler)

        Parameters
        ----------
        P_mvdr : np.ndarray
            MVDR 功率譜矩陣, shape = (len(theta), len(freqs))
        theta : np.ndarray
            AoA 角度軸 (degrees)
        freqs : np.ndarray
            Doppler 軸 (Hz or normalized frequency)
        top_k : int, optional
            要回傳的前幾個最大峰值 (default=1)
        threshold_ratio : float, optional
            過濾小於主峰 x% 的次峰 (default=0.5, 表主峰一半以下的忽略)

        Returns
        -------
        peaks_info : list of dict
            每個峰包含 {'theta': ..., 'freq': ..., 'value': ...}
        """

        # 1️⃣ 找出最大值與索引
        P_abs = np.abs(P_mvdr)
        max_val = np.max(P_abs)

        # 2️⃣ 設定門檻 (防止雜訊假峰)
        mask = P_abs >= threshold_ratio * max_val
        # 3️⃣ 在 2D 矩陣中找局部峰
        peaks = []
        for i in range(1, P_abs.shape[0]-1):
            for j in range(1, P_abs.shape[1]-1):
                if mask[i, j] and P_abs[i, j] == np.max(P_abs[i-1:i+2, j-1:j+2]):
                    peaks.append((i, j, P_abs[i, j]))

        # 4️⃣ 排序取 top_k
        peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:top_k]

        # 5️⃣ 轉換成 AoA–Doppler 對應值
        peaks_info = [
            {'theta': theta[i], 'freq': freqs[j], 'value': val}
            for i, j, val in peaks
        ]
        print(peaks_info)

        return peaks_info

    def find_AoA_ToF_peaks(P_music, theta, tau, num_peaks=2, sigma=2, neighborhood_size=5, threshold=0.3):
            # Normalize to 0~1 for uniform comparison
            P_norm = (P_music - np.min(P_music)) / (np.max(P_music) - np.min(P_music) + 1e-12)

            # 1️⃣ Gaussian smoothing (optional but helps suppress noise)
            P_smooth = gaussian_filter(P_norm, sigma=sigma)

            # 2️⃣ Local maxima detection
            local_max = (P_smooth == maximum_filter(P_smooth, size=neighborhood_size))

            # 3️⃣ Threshold filtering
            mask = local_max & (P_smooth >= threshold)
            peak_indices = np.argwhere(mask)

            # 4️⃣ Sort by power (descending)
            sorted_idx = np.argsort(P_smooth[mask].ravel())[::-1]
            peak_indices = peak_indices[sorted_idx]

            # 5️⃣ Select top-N peaks
            peaks = []
            for idx in peak_indices[:num_peaks]:
                i, j = idx
                peaks.append({
                    'theta': float(theta[i]),
                    'tau': float(tau[j])*1e9,
                    'power': float(P_smooth[i, j]),
                    'index': (int(i), int(j))
                })
            for peak in peaks:
                print(f"theta: {peak['theta']:>7.3f}   tau: {peak['tau']:>7.3f}   power: {peak['power']:>7.3f}")
            print('\n')
            return peaks

    def find_peak_aoa_doppler_v2(P_mvdr, theta, freqs, top_k=2, threshold_ratio=0.1):
        P_abs = np.abs(P_mvdr)
        max_val = np.max(P_abs)
        mask = P_abs >= threshold_ratio * max_val

        peaks = []
        for i in range(P_abs.shape[0]):
            for j in range(P_abs.shape[1]):
                r0, r1 = max(0, i-1), min(P_abs.shape[0], i+2)
                c0, c1 = max(0, j-1), min(P_abs.shape[1], j+2)
                if mask[i, j] and P_abs[i, j] >= np.max(P_abs[r0:r1, c0:c1]) * 0.98:
                    peaks.append((i, j, P_abs[i, j]))
                    

        peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:top_k]

        peaks_info = [
            {'theta': theta[i], 'freq': freqs[j], 'value': val}
            for i, j, val in peaks
        ]
        return peaks_info

class AoA_ToF:
    def AoA_ToF_estimator(CSI, subc_stride, stream_window_size, subc_window_size, args, avg=False, temp_smoothed=False, ground_truth=None):
        if temp_smoothed:
            N_temp = 2   
            temp_list = []   

            for i in range(-N_temp, N_temp+1):
                smoothed_csi = AoA_ToF.smoothed_csi(
                    CSI[args.plotted_packet + i],
                    args,
                    stream_window_size,
                    subc_window_size,
                    subc_stride
                    )
                temp_list.append(smoothed_csi)
            temp_smoothed_CSIs = np.hstack(temp_list)
            print(f"[Temp smoothed] N_temp={N_temp}, frames={2*N_temp+1}, shape={temp_smoothed_CSIs.shape}")
        elif avg:
            smoothed_CSIs = []
            for i in range(args.nperseg):
                smoothed_csi = AoA_ToF.smoothed_csi(
                    CSI[args.plotted_packet -args.nperseg//2 + i], 
                    args, 
                    stream_window_size, 
                    subc_window_size, 
                    subc_stride
                    )
                smoothed_CSIs.append(smoothed_csi)
            smoothed_CSIs = np.array(smoothed_CSIs)
            print(f"[smoothed avg] avg frames={args.nperseg}, shape={smoothed_CSIs.shape}")
        else:
            smoothed_CSI = AoA_ToF.smoothed_csi(
                CSI[args.plotted_packet], 
                args, 
                stream_window_size, 
                subc_window_size, 
                subc_stride
                )
            print(f"[smoothed], shape={smoothed_CSI.shape}")
        input = (temp_smoothed_CSIs if temp_smoothed else smoothed_CSIs if avg else smoothed_CSI)
            
        tau, theta, P_music = AoA_ToF.cal_AoA_ToF(
            input,
            args,
            stream_window_size, 
            subc_window_size, 
            subc_stride
            )
        AoA_ToF.plot_AoA_ToF(
            tau, 
            theta, 
            P_music, 
            title= f"(temp smoothed) N={N_temp}" if temp_smoothed else "(smoothed avg)" if avg else "(smoothed)",
            ground_truth=ground_truth
        )
        pass

    def smoothed_csi(csi, args, stream_window_size, subc_window_size, subc_stride=1):
        """
        Build Toeplitz-like smoothed CSI for AoA–ToF. (Use Tx=0)
        """
        # block_size = num_subc_per_block
        block_size = subc_window_size // subc_stride

        H_list = []
        for r in range(args.num_Rx):
            H = np.zeros((block_size, block_size), dtype=complex)
            for i in range(block_size):
                start = i * subc_stride
                end = min(start + block_size, args.num_subcarriers)
                start = end - block_size
                H[i, :] = csi[0, r, start:end]
            H_list.append(H)

        Ncol = stream_window_size if (args.num_Rx % 2 == 1) else (stream_window_size - 1)

        smoothed_csi = np.block([
            [H_list[i + j] for j in range(Ncol)]
            for i in range(stream_window_size)
        ])

        return smoothed_csi

    def cal_AoA_ToF(x, args, stream_window_size, subc_window_size, subc_stride=1):
        #print("smoothed shape:", x.shape)
        # Covariance matrix
        if len(x.shape) == 2:
            x = np.asarray(x, dtype=complex)
            S = x @ x.conj().T
        elif len(x.shape) == 3:
            S = 0
            for i in range(x.shape[0]):
                temp_x = np.asarray(x[i], dtype=complex)
                S += temp_x @ temp_x.conj().T
            S = S / x.shape[0]

        # Eigen decomposition
        eig_val, eig_vec = np.linalg.eigh(S)
        eig_vec = eig_vec.astype(complex)
        idx_order = eig_val.argsort()[::-1]
        eig_val = eig_val[idx_order]
        eig_vec = eig_vec[:, idx_order]

        # Noise subspace
        Sdim = 11
        N_dim = eig_val.shape[0] - Sdim
        E_n = eig_vec[:, -N_dim:]
        P_n = E_n @ E_n.conj().T

        # theta candidate
        theta = np.arange(0, 360) if args.projection=='sin' else np.arange(0, 180)
        # tau candidate
        tau = np.arange(0.2*1e-8, 3.3*1e-8, 5e-10) #620 points

        # steering_vector length:
        sv_len = stream_window_size * (subc_window_size // subc_stride)
        # calculate all steering vectors at once:
        Steering_Vectors = np.zeros((len(theta), len(tau), sv_len), dtype=complex)
        for i in range(len(theta)):
            for j in range(len(tau)):
                Steering_Vectors[i,j,:] = AoA_ToF.steering_vector_AoA_ToF(
                    theta_i = theta[i],
                    tau_j = tau[j],
                    args = args,
                    stream_window_size = stream_window_size,
                    subc_window_size = subc_window_size,
                    subc_stride = subc_stride
                )

        # MUSIC spectrum
        #Pn = EnEn^H and PP = s^T @ Pn @ s = s^T @ EnEn^H @ s
        #let a = s^T @ En and PP = a^* @ a = |a|^2

        # 1)
        SV_flat = Steering_Vectors.reshape(len(theta) * len(tau), sv_len)
        # SV_flat 的第 p 列，就是某一組 (θ_i, τ_j) 的 steering vector：
        # p = i * num_tau + j 對應 (i, j)

        # 2) 投影到 noise subspace: (T*K, N_dim)
        A = SV_flat @ E_n     # E_n: (N, N_dim)

        # 3) 每個 (θ,τ) 的分母：‖E_n^H s‖²
        PP_flat = np.sum(np.abs(A)**2, axis=1)

        # 4) reshape 回 (θ, τ)
        PP = PP_flat.reshape(len(theta), len(tau))

        # 5) MUSIC spectrum
        P_music = 10 * np.log10(1.0 / PP)

        return tau, theta, P_music

    def steering_vector_AoA_ToF(theta_i, tau_j, args, stream_window_size, subc_window_size, subc_stride=1):

        fc = args.f_0
        delta_f = args.delta_f
        theta_i = np.deg2rad(theta_i)
        
        # ----------------------
        # Subcarrier (ToF + carrier) phase
        # ----------------------
        row_size = subc_window_size // subc_stride  # number of subcarriers after stride
        sub_idx = np.arange(0, row_size) * subc_stride   # indices of selected subcarriers

        # carrier delay
        const_phase = np.exp(-2j * np.pi * fc * tau_j)
        # subcarrier frequency phase
        exp_omega = const_phase * np.exp(-2j * np.pi * delta_f * tau_j * sub_idx)

        # ----------------------
        # Spatial phase (AoA)
        # ----------------------
        if args.projection == "sin":
            exp_phi = np.exp(+2j * np.pi * fc * np.sin(theta_i) * args.d / 3e8 * np.arange(stream_window_size))
        elif args.projection == "cos":
            exp_phi = np.exp(2j * np.pi * fc * (1 - np.cos(theta_i)) * args.d / 3e8 * np.arange(stream_window_size))
        
        # ----------------------
        # Steering vector = Kronecker of spatial and subcarrier vectors
        # ----------------------
        steering_vector = np.kron(exp_phi, exp_omega)
        return steering_vector.flatten()

    def plot_AoA_ToF(tau, theta, P_music, title="", ground_truth={}):
        #print(title)
        peaks = find_Peaks.find_AoA_ToF_peaks(P_music, theta, tau)
        plt.figure()
        plt.pcolormesh(tau, theta, P_music, cmap = 'jet', shading = 'auto')
        plt.colorbar()
        for _, (theta_gt, tau_gt) in ground_truth.items():
            plt.scatter(tau_gt, theta_gt, color='black', marker='x', s=50)
        plt.xlabel('tau (s)')
        plt.ylabel('theta (deg)')
        plt.title('AoA-ToF MUSIC '+title, fontsize = 10)

    def FB_smooth(x, args, subc_window_size, subc_stride):
        """
        Forward + Backward smoothing for multi-Rx CSI
        x: CSI[Tx=0], shape = (1, num_Rx, 512)
        """
        block_size = subc_window_size // subc_stride
        num_blocks = (args.num_subcarriers - subc_window_size) // subc_stride + 1

        # ---------- Forward ----------
        H_forward = []
        for i in range(num_blocks):
            start = i * subc_stride
            end = start + subc_window_size

            Fi = []
            for r in range(args.num_Rx):
                xr = x[0, r, :]
                window = xr[start:end]
                Fi_r = window[0:block_size * subc_stride:subc_stride]
                Fi.append(Fi_r.astype(complex))

            H_forward.append(np.hstack(Fi))

        H_forward = np.array(H_forward, dtype=complex)

        # ---------- Backward ----------
        H_backward = []
        for i in range(num_blocks):
            start = i * subc_stride

            Bi = []
            for r in range(args.num_Rx):
                xr = x[0, r, :]
                xb = np.conj(xr[::-1])
                window = xb[start:start + subc_window_size]
                Bi_r = window[0:block_size * subc_stride:subc_stride]
                Bi.append(Bi_r.astype(complex))

            H_backward.append(np.hstack(Bi))

        H_backward = np.array(H_backward, dtype=complex)

        # ---------- Merge ----------
        H_FB = np.vstack([H_forward, H_backward])  # (2*num_blocks, 320)

        return H_FB.T
    
    def FB_cal_AoA_ToF(x, args, subc_window_size, subc_stride=1):
        #print('Forward + Backward smoothed shape: ', x.shape)
        # Covariance matrix
        if len(x.shape) == 2:
            x = np.asarray(x, dtype=complex)
            S = x @x.conj().T
        elif len(x.shape) == 3:
            S = 0
            for i in range(x.shape[0]):
                temp_x = np.asarray(x[i], dtype=complex)
                S += temp_x @ temp_x.conj().T
            S = S / x.shape[0]

        # Eigen decomposition
        eig_val, eig_vec = np.linalg.eigh(S)
        eig_vec = eig_vec.astype(complex)
        idx_order = eig_val.argsort()[::-1]
        eig_val = eig_val[idx_order]
        eig_vec = eig_vec[:, idx_order]

        # Noise subspace
        Sdim = 10
        N_dim = eig_val.shape[0] - Sdim
        E_n = eig_vec[:, -N_dim:]
        P_n = E_n @ E_n.conj().T

        # theta candidate
        theta = np.arange(-90, 90) if args.projection=='sin' else np.arange(0, 180)
        # tau candidate
        tau = np.arange(0.2*1e-8, 3.3*1e-8, 5e-10) #620 points

        # steering_vector length:
        sv_len = args.num_Rx * (subc_window_size // subc_stride)
        # calculate all steering vectors at once:
        Steering_Vectors = np.zeros((len(theta), len(tau), sv_len), dtype=complex)
        for i in range(len(theta)):
            for j in range(len(tau)):
                Steering_Vectors[i,j,:] = AoA_ToF.steering_vector_AoA_ToF(
                    theta_i = theta[i],
                    tau_j = tau[j],
                    args = args,
                    stream_window_size= args.num_Rx,
                    subc_window_size = subc_window_size,
                    subc_stride = subc_stride
                )


        # MUSIC spectrum
        #Pn = EnEn^H and PP = s^T @ Pn @ s = s^T @ EnEn^H @ s
        #let a = s^T @ En and PP = a^* @ a = |a|^2

        # 1)
        SV_flat = Steering_Vectors.reshape(len(theta) * len(tau), sv_len)
        # SV_flat 的第 p 列，就是某一組 (θ_i, τ_j) 的 steering vector：
        # p = i * num_tau + j 對應 (i, j)

        # 2) 投影到 noise subspace: (T*K, N_dim)
        A = SV_flat @ E_n     # E_n: (N, N_dim)

        # 3) 每個 (θ,τ) 的分母：‖E_n^H s‖²
        PP_flat = np.sum(np.abs(A)**2, axis=1)

        # 4) reshape 回 (θ, τ)
        PP = PP_flat.reshape(len(theta), len(tau))

        # 5) MUSIC spectrum
        P_music = 10 * np.log10(1.0 / PP)

        return tau, theta, P_music
    
    def FB_AoA_ToF_estimator(CSI, subc_stride, subc_window_size, args, avg=False, temp_smoothed=False, ground_truth=None):
        if temp_smoothed:
            N_temp = 10   
            temp_list = []   

            for i in range(-N_temp, N_temp+1):
                smoothed_csi = AoA_ToF.FB_smooth(
                    CSI[args.plotted_packet + i],
                    args,
                    subc_window_size,
                    subc_stride
                )
                temp_list.append(smoothed_csi)
            temp_smoothed_CSIs = np.hstack(temp_list)
            print(f"[Forward + Backward Temp smoothed] N_temp={N_temp}, frames={2*N_temp+1}, shape={temp_smoothed_CSIs.shape}")
        elif avg:
            smoothed_CSIs = []
            for i in range(args.nperseg+2):
                smoothed_csi = AoA_ToF.FB_smooth(
                    CSI[args.plotted_packet -args.nperseg//2 + i], 
                    args, 
                    subc_window_size, 
                    subc_stride
                    )
                smoothed_CSIs.append(smoothed_csi)
            smoothed_CSIs = np.array(smoothed_CSIs)
            print(f"[Forward + Backward avg] avg frames={args.nperseg}, shape={smoothed_CSIs.shape}")
        else:
            smoothed_CSI = AoA_ToF.FB_smooth(
                CSI[args.plotted_packet], 
                args, 
                subc_window_size, 
                subc_stride
                )
            print(f"[Forward + Backward], shape={smoothed_CSI.shape}")
 
        input = (temp_smoothed_CSIs if temp_smoothed else smoothed_CSIs if avg else smoothed_CSI)

        #covariance_plot(abs(input))

        tau, theta, P_music = AoA_ToF.FB_cal_AoA_ToF(
            input,
            args,
            subc_window_size,
            subc_stride
        )

        AoA_ToF.plot_AoA_ToF(
            tau,
            theta,
            P_music,
            title= f"(FB temp smoothed) N={N_temp}" if temp_smoothed else "(FB avg frame)" if avg else "(FB single frame)",
            ground_truth=ground_truth
        )
    
class Reference:
    def smoothed_csi_AoA_ToF(csi, args, stream_window_size, subc_window_size, subc_stride=1):
        """
        Build Toeplitz-like smoothed CSI for AoA–ToF. (Use Tx=0)
        """
        # block_size = num_subc_per_block
        block_size = subc_window_size // subc_stride

        H_list = []
        for r in range(args.num_Rx):
            H = np.zeros((block_size, block_size), dtype=complex)
            for i in range(block_size):
                start = i * subc_stride
                end = min(start + block_size, args.num_subcarriers)
                start = end - block_size
                H[i, :] = csi[0, r, start:end]
            H_list.append(H)

        Ncol = stream_window_size if (args.num_Rx % 2 == 1) else (stream_window_size - 1)

        smoothed_csi = np.block([
            [H_list[i + j] for j in range(Ncol)]
            for i in range(stream_window_size)
        ])

        return smoothed_csi

    def smoothed_csi_AoA_ToFX(csi, args, stream_window_size, subc_window_size, subc_stride=1):
        num_Rx = args.num_Rx
        num_subcarriers = args.num_subcarriers
        row_size = subc_window_size // subc_stride  # 每個 H 的行數
        """
        Generate Toeplitz-like smoothed CSI with adjustable subcarrier stride.
        
        Parameters
        ----------
        csi : np.ndarray
            Shape = (num_Tx, num_Rx, num_subcarriers)
            Only uses the first Tx: csi[0]
        stream_window_size : int
            Number of Hankel matrices in row (time/stream direction)
        subc_window_size : int
            Size of subcarrier window (before stride)
        subc_stride : int
            Step size for subcarrier sampling

        Returns
        -------
        smoothed_csi : np.ndarray
            Toeplitz-like smoothed CSI matrix
            Shape = (stream_window_size*(subc_window_size//subc_stride),
                    stream_window_size*(subc_window_size//subc_stride))
        """
        H_list = []
        for r in range(num_Rx):
            H = np.zeros((row_size, row_size), dtype=complex)
            for i in range(row_size):
                start_idx = i * subc_stride
                end_idx = start_idx + row_size
                if end_idx > num_subcarriers:
                    end_idx = num_subcarriers
                    start_idx = end_idx - row_size
                H[i, :] = csi[0, r, start_idx:end_idx]
            H_list.append(H)
        
        Ncol = stream_window_size if num_Rx % 2 == 1 else stream_window_size - 1
        
        smoothed_csi = np.block([[H_list[i + j] for j in range(Ncol)] for i in range(stream_window_size)])
        return smoothed_csi

    def steering_vector_AoA_ToF(theta_i, tau_j, args, stream_window_size, subc_window_size, subc_stride=1):

        fc = args.f_0
        delta_f = args.delta_f
        theta_i = np.deg2rad(theta_i)
        
        # ----------------------
        # Subcarrier (ToF + carrier) phase
        # ----------------------
        row_size = subc_window_size // subc_stride  # number of subcarriers after stride
        sub_idx = np.arange(0, row_size) * subc_stride   # indices of selected subcarriers

        # carrier delay
        const_phase = np.exp(-2j * np.pi * fc * tau_j)
        # subcarrier frequency phase
        exp_omega = const_phase * np.exp(-2j * np.pi * delta_f * tau_j * sub_idx)

        # ----------------------
        # Spatial phase (AoA)
        # ----------------------
        if args.projection == "sin":
            exp_phi = np.exp(+2j * np.pi * fc * np.sin(theta_i) * args.d / 3e8 * np.arange(stream_window_size))
        elif args.projection == "cos":
            exp_phi = np.exp(2j * np.pi * fc * (1 - np.cos(theta_i)) * args.d / 3e8 * np.arange(stream_window_size))
        
        # ----------------------
        # Steering vector = Kronecker of spatial and subcarrier vectors
        # ----------------------
        steering_vector = np.kron(exp_phi, exp_omega)
        return steering_vector.flatten()

    def cal_AoA_ToF(x, args, stream_window_size, subc_window_size, subc_stride=1):

        # Covariance matrix
        if len(x.shape) == 2:
            x = np.asarray(x, dtype=complex)
            S = x @ x.conj().T
        elif len(x.shape) == 3:
            S = 0
            for i in range(x.shape[0]):
                temp_x = np.asarray(x[i], dtype=complex)
                S += temp_x @ temp_x.conj().T
            S = S / x.shape[0]

        # Eigen decomposition
        eig_val, eig_vec = np.linalg.eigh(S)
        eig_vec = eig_vec.astype(complex)
        idx_order = eig_val.argsort()[::-1]
        eig_val = eig_val[idx_order]
        eig_vec = eig_vec[:, idx_order]

        # Noise subspace
        Sdim = 11
        N_dim = eig_val.shape[0] - Sdim
        E_n = np.asarray(eig_vec[:, -N_dim:])
        E_n = np.asarray(E_n)
        P_n = E_n @ E_n.conj().T

        # theta candidate
        theta = np.arange(-90, 90) #if args.projection=='sin' else np.arange(0, 180)
        # tau candidate
        tau = np.arange(0.2*1e-8, 3.3*1e-8, 5e-10) #620 points

        # MUSIC spectrum
        P_music = np.zeros((len(theta), len(tau)))
        for i in range(len(theta)):
            for j in range(len(tau)):
                # 使用 stride 版本的 steering vector
                steering_vector = Reference.steering_vector_AoA_ToF(
                    theta_i = theta[i],
                    tau_j = tau[j],
                    args = args,
                    stream_window_size = stream_window_size,
                    subc_window_size = subc_window_size,
                    subc_stride = subc_stride
                )
                PP = steering_vector.conj().T @ P_n @ steering_vector
                P_music[i,j] = 10 * np.log10(abs(1 / PP))

        return tau, theta, P_music


    def plot_AoA_ToF(tau, theta, P_music, title=""):
        
        plt.figure()
        plt.pcolormesh(tau, theta, P_music, cmap = 'jet', shading = 'auto')
        plt.colorbar()
        plt.xlabel('tau (s)')
        plt.ylabel('theta (deg)')
        plt.title('AoA-ToF MUSIC'+title, fontsize = 10)

    def sampled_CSI_AoA_Doppler(CSI, args, tx_id = 0,subcarrier_id = 0): 
        # sampled_csi[time_id - nperseg ~ time_id][0][i][0] 
        # # focus on the moving targets varying along time 
        sampled_CSI = np.zeros((args.nperseg, args.num_Rx), dtype=np.complex64) 
        for i in range(args.nperseg): 
            for rx in range(args.num_Rx): 
                sampled_CSI[i, rx] = CSI[args.plotted_packet + i, tx_id, rx, subcarrier_id] 
        return sampled_CSI

    def cal_AoA_Doppler_MUSIC(x, args):
        if len(x.shape) == 2:
            x = np.matrix(x, dtype = complex)
            R = x @ x.conj().T

        elif len(x.shape) == 3:
            R = 0
            for i in range(x.shape[0]):
                temp_x = np.matrix(x[i])
                R += np.matmul(temp_x, temp_x.H)
            R = R / x.shape[0]
        
        # calculate eigenvalues and eigenvectors
        eig_val, eig_vec = np.linalg.eigh(R)  
        eig_val = np.abs(eig_val)
        eig_vec = eig_vec.astype(complex)

        # sort the eigenvalues and eigenvectors
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx] #eig_val.shape: (768,)
        eig_vec = eig_vec[:, idx]
        ### TRY optimize idx ###
        idx = 10

        # determine the dimension of the signal space and the noise space
        S_dim = idx + 1 
        N_dim = eig_val.shape[0] - (idx + 1)

        E_n = eig_vec[:, -N_dim:]
        P_n = E_n @ E_n.conj().T
        
        if args.projection == 'sin':
            theta = np.arange(-90, 91)
        elif args.projection == 'cos':
            theta = np.arange(0, 180, 1)
        
        # doppler frequency candidate
        freqs = np.arange(-40, 40, 1)
        
        # calculate P_music
        #P_music(freq, theta) = 1/a^H E_n E_n^H a
        P_music = np.zeros([theta.shape[0], freqs.shape[0]])
        for i in range(theta.shape[0]):
            for j in range(freqs.shape[0]):		
                steering_vector = Reference.steering_vector_AoA_Doppler_MUSIC(theta[i], freqs[j], args)
                #reshape the steering vector
                steering_vector = np.reshape(steering_vector, [steering_vector.shape[0], 1])
                # caculate P_music in dB 
                PP = steering_vector.conj().T @ P_n @ steering_vector
                P_music[i, j] = 10 * np.log10(abs(1 / PP))
        #⚠️不知道為什麼圖是反的
        # 翻轉    
        P_music = np.flip(P_music, axis = 0)
        
        return freqs, theta, P_music

    def steering_vector_AoA_Doppler_MUSIC(theta_i, f_j, args):
        #steering vector elements
        #(1) phase shift by AoA --------- e^(-j 2π sin(θi) d / λ ), λ = c / fc
        #(2) phase rotation by dopler --- e^(+ j 2π fj / fs)

        theta_i = np.deg2rad(theta_i) 

        #vector exp_phi (AoA term)
        #[ 1, e^jφ, e^j2φ, e^j3φ, ...e^j(Nstream -1)φ ]
        if args.projection == "sin":
            exp_phi = np.exp( -2j * np.pi * args.f_0 * np.sin(theta_i) * args.d / 3e8 * np.arange(args.num_Rx))
        elif args.projection == "cos":  
            exp_phi = np.exp(  2j * np.pi * args.f_0 * (1 - np.cos(theta_i)) * args.d / 3e8 * np.arange(args.num_Rx))
        #vector exp_phi (doppler term)
        #[ 1, e^jω, e^j2ω, e^j3ω, ...e^j(Nstream * Nsubc -1)φ ]
        exp_omega = np.exp(1j * 2 * np.pi * np.arange(0, args.nperseg) * f_j / args.fs)
        #steering vector
        #[ 1     * [1, e^jω, e^j2ω, e^j3ω, ...e^j(num_subcarriers-1)ω] ]
        #| e^jφ  * [1, e^jω, e^j2ω, e^j3ω, ...e^j(num_subcarriers-1)ω] |
        #[ e^j2φ * [1, e^jω, e^j2ω, e^j3ω, ...e^j(num_subcarriers-1)ω] ]Nstream x (Nstream*Nsubc)
        steering_vector = np.kron(exp_phi, exp_omega)
        #reshape to 1D array
        return steering_vector.reshape(-1, 1)

    def smooth_CSI_AoA_f_doppler(CSI, args):
        num_streams = args.num_Rx
        nperseg = args.nperseg
        sub_nperseg = args.sub_nperseg
        """
        Spatial–temporal smoothing for AoA–Doppler estimation.

        Input:
            CSI.shape = (nperseg, num_streams)
        Output:
            smoothed_CSI.shape = (sub_nperseg * num_segments,
                                (nperseg - sub_nperseg) * num_segments)
        """
        num_segments = num_streams // 2 + 1

        segments = np.array([
            [CSI[j:j + (nperseg - sub_nperseg), i] for j in range(sub_nperseg)]
            for i in range(num_streams)
        ], dtype=complex)  


        smoothed_CSI = np.zeros(
            (sub_nperseg * num_segments, (nperseg - sub_nperseg) * num_segments),
            dtype=complex
        )
        for p in range(num_segments):
            for q in range(num_segments):
                if p + q < num_streams:  
                    smoothed_CSI[
                        p * sub_nperseg:(p + 1) * sub_nperseg,
                        q * (nperseg - sub_nperseg):(q + 1) * (nperseg - sub_nperseg)
                    ] = segments[p + q]
        return smoothed_CSI

    def cal_AoA_Doppler_smoothed(x, args):

        if len(x.shape) == 2:
            x = np.matrix(x, dtype = complex)
            R = x @ x.conj().T

        elif len(x.shape) == 3:
            R = 0
            for i in range(x.shape[0]):
                temp_x = np.matrix(x[i])
                R += np.matmul(temp_x, temp_x.H)
            R = R / x.shape[0]
        
        # calculate eigenvalues and eigenvectors
        eig_val, eig_vec = np.linalg.eigh(R)  
        eig_val = np.abs(eig_val)
        eig_vec = eig_vec.astype(complex)

        # sort the eigenvalues and eigenvectors
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx] #eig_val.shape: (768,)
        eig_vec = eig_vec[:, idx]
        ### TRY optimize idx ###
        idx = 10

        # determine the dimension of the signal space and the noise space
        S_dim = idx + 1 
        N_dim = eig_val.shape[0] - (idx + 1)

        E_n = eig_vec[:, -N_dim:]
        P_n = E_n @ E_n.conj().T
        
        if args.projection == 'sin':
            theta = np.arange(-90, 91)
        elif args.projection == 'cos':
            theta = np.arange(0, 180, 1)
        
        # doppler frequency candidate
        freqs = np.arange(-30, 80, 1)
        
        # calculate P_music
        #P_music(freq, theta) = 1/a^H E_n E_n^H a
        P_music = np.zeros([theta.shape[0], freqs.shape[0]])
        for i in range(theta.shape[0]):
            for j in range(freqs.shape[0]):		
                steering_vector = Reference.steering_vector_AoA_Doppler_smoothed(theta[i], freqs[j], args)
                #reshape the steering vector
                steering_vector = np.reshape(steering_vector, [steering_vector.shape[0], 1])
                # caculate P_music in dB 
                PP = steering_vector.conj().T @ P_n @ steering_vector
                P_music[i, j] = 10 * np.log10(abs(1 / PP))
        #⚠️vertical filp
        P_music = np.flipud(P_music)
        
        return freqs, theta, P_music
    
    def steering_vector_AoA_Doppler_smoothed(theta_i, f_j, args):
        # convert to radian
        theta = np.deg2rad(theta_i)

        num_stream = (args.num_Rx+1)//2
        sub_nperseg = args.sub_nperseg
        steering_vector = np.zeros((sub_nperseg * num_stream, 1), dtype = complex)

        # light speed
        c = 3e8
        if args.projection == 'sin':
            spatial_phase = np.exp(-1j * 2 * np.pi * args.f_0 * args.d * np.sin(theta) / c)
        elif args.projection == 'cos':
            spatial_phase = np.exp(1j * 2 * np.pi * args.f_0 * args.d * (1 - np.cos(theta)) / c)
        else:
            raise ValueError("args.projection must be 'sin' or 'cos'")

        base = 1.0 + 0j
        for i in range(num_stream):
            for j in range(sub_nperseg):
                idx = i * sub_nperseg + j
                steering_vector[idx] = base * np.exp(1j * 2 * np.pi * f_j * j / args.fs)
            base *= spatial_phase  # move to next antenna

        return steering_vector

    def plot_AoA_Doppler(freq, theta, P_music, title=""):
        plt.figure()
        plt.pcolormesh(freq, theta, P_music, cmap = 'jet', shading = 'auto')
        plt.colorbar()
        plt.xlabel('Doppler (Hz)')
        plt.ylabel('theta (deg)')
        plt.title('AoA-Doppler '+title, fontsize = 10)
        #AUTO_SAVE(plt.gcf(), f"AoA_Doppler_MUSIC_{title}")

    def cal_AoA_Doppler_beamform(x, args):
        if len(x.shape) == 2:
            #x(num_samples, num_Rx) = (50, 5)
            R = np.zeros((args.num_Rx, args.num_Rx), dtype=complex)
            for j in range(x.shape[0]):
                x_col = x[j, :].reshape(-1, 1)   # reshape to column vector
                R += x_col @ x_col.conj().T #sum of time samples
            eps = 1e-3
            R = R / x.shape[0] + eps * np.eye(args.num_Rx)
            R_inv = np.linalg.inv(R)
            """
        elif len(x.shape) == 3:
            # x(num_subc_sample, num_time_sample, num_Rx) = (10, 50, 5)
            R_avg = np.zeros((args.num_Rx, args.num_Rx), dtype=complex)  # Initialize the average correlation matrix
            for i in range(x.shape[0]):  # Loop through subcarrier samples
                R = np.zeros((args.num_Rx, args.num_Rx), dtype=complex)  # Initialize the correlation matrix for each subcarrier
                for j in range(x.shape[1]):  # Loop through each time sample
                    x_col = x[i, j, :].reshape(-1, 1)  # Reshape to a column vector
                    R += x_col @ x_col.conj().T  # Accumulate the outer product
                eps = 1e-3
                R = R / x.shape[1] + eps * np.eye(args.num_Rx)  # Time-averaged covariance matrix with regularization
                # Calculate the inverse of R for each subcarrier
                R_inv = np.linalg.inv(R)
                # Accumulate the inverse of R for all subcarriers
                R_avg += R_inv
            

            # Average the inverses of all subcarriers
            R_avg = R_avg / x.shape[0]  # Average over subcarriers
            x = x[x.shape[0] // 2, :, :]  # Use the first subcarrier's CSI data for beamforming
            """
            
        elif len(x.shape) == 3:
            # x(num_subc_sample, num_time_sample, num_Rx) = (10, 50, 5)
            R_avg = np.zeros((args.num_Rx, args.num_Rx), dtype=complex)  # Initialize the average correlation matrix

            # Loop through subcarrier samples and accumulate covariance matrix
            for i in range(x.shape[0]):  # Loop through subcarrier samples
                R = np.zeros((args.num_Rx, args.num_Rx), dtype=complex)  # Initialize the correlation matrix for each subcarrier
                
                # Loop through time samples and accumulate outer products
                for j in range(x.shape[1]):  # Loop through each time sample
                    x_col = x[i, j, :].reshape(-1, 1)  # Reshape to a column vector
                    R += np.outer(x_col, x_col.conj())  # Efficiently accumulate the outer product

                eps = 1e-3
                R = R / x.shape[1] + eps * np.eye(args.num_Rx)  # Time-averaged covariance matrix with regularization
                
                # Accumulate the covariance matrices (no need to calculate inverse here)
                R_avg += R

            # Average the covariance matrices over all subcarriers
            R_avg = R_avg / x.shape[0]  # Average over subcarriers

            # Calculate the inverse of the averaged covariance matrix
            R_inv = np.linalg.inv(R_avg)  # Inverse of the averaged covariance matrix

            # Optionally use the first subcarrier's CSI data for beamforming (if needed)
            x = x[x.shape[0] // 2, :, :]  # Use the first subcarrier's CSI data for beamforming
    
        # x = sampled_csi #shape(50,5)
        # (1) R = 1/T ∑ x x^H
        # min w^H x R x w subject to w^H a(θ) = 1
        # (2) w_opt = R^(-1) a(θ) / a(θ)^H R^-1 a(θ)

        # (3) y = w_opt^H x
        #  P_mvdr(θ) = 1/a(θ)^H R^-1 a(θ)
        # (4) P_mvdr(θ, f) = | ∑ y[j]exp(j 2 pi k*fd / fs) |
        # theata and frenquency candidate

        if args.projection == "sin": theta = np.arange(-90, 91)
        elif args.projection == "cos": theta = np.arange(0, 181)
        freqs = np.arange(-25, 70)
        
        P_mvdr = np.zeros([theta.shape[0], freqs.shape[0]])

        for i in range(theta.shape[0]):
            steering_vector = Reference.steering_vector_AoA_Doppler_beamform(theta[i], args)
            steering_vector = np.matrix(steering_vector.reshape(-1, 1))
            """
            R = np.zeros((args.num_Rx, args.num_Rx),dtype=complex)

            #(1) R = 1/T ∑ x[θi]x[θi]^H

            for j in range(x.shape[0]):
                x_col = x[j, :].reshape(-1, 1)   # reshape to column vector
                R += x_col @ x_col.conj().T
            eps = 1e-3
            R = R / x.shape[0] + eps * np.eye(args.num_Rx)
            R_inv = np.linalg.inv(R)

            """


            #(2) W_opt = R^(-1) a(θ) / a(θ)^H R^-1 a(θ)
            W_opt = R_inv @ steering_vector / np.matmul(steering_vector.conj().T, R_inv @ steering_vector)
            #print("W_opt shape:", W_opt.shape) #(5,1)
            #print("x shape:", x.shape) #(50,5)

            #(3) project x onto the W_opt
            beamformed_x = (W_opt.conj().T @ x.T).T 

            # P_mvdr(θ, f) = | ∑ beamformed_x(θ)[j] * e^(-j 2π k f / fs) |

            for j in range(freqs.shape[0]):
                f_bin = 0
                for k in range(x.shape[0]):
                    f_bin += beamformed_x[k] * np.exp(-1j * 2 * np.pi * k * freqs[j] / args.fs)
                P_mvdr[i, j] = abs(f_bin)
            
            
        return freqs, theta, P_mvdr

    def steering_vector_AoA_Doppler_beamform(theta_i, args):
        steering_vector = np.zeros((args.num_Rx, 1), dtype = complex)
        theta = np.deg2rad(theta_i)
        c = 3e8
        base = 1
        for i in range(args.num_Rx):
            steering_vector[i] = base
            if args.projection == 'sin':
                base = base * np.exp(-1j * 2 * np.pi * args.f_0 * args.d * np.sin(theta) / c)
            elif args.projection == 'cos':
                base = base * np.exp(1j * 2 * np.pi * args.f_0 * args.d * (1 - np.cos(theta)) / c)

        return steering_vector.reshape(-1, 1) #column vector
