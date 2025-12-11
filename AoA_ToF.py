import numpy as np
import matplotlib.pyplot as plt
import AoA_ToF 
from scipy.ndimage import gaussian_filter, maximum_filter


def evaluate_AoA_ToF_methods(args, CSI, ground_truth):
    methods = {
        'smoothed':         (AoA_ToF.AoA_ToF_estimator, 1, False, False),
        'smoothed_avg':     (AoA_ToF.AoA_ToF_estimator, 4, True, False),
        'temp_smoothed':    (AoA_ToF.AoA_ToF_estimator, 1, False, True),
        'FB_smoothed':      (AoA_ToF.FB_AoA_ToF_estimator, 1, False, False),
        'FB_smoothed_avg':  (AoA_ToF.FB_AoA_ToF_estimator, 4, True, False),
        'FB_temp_smoothed': (AoA_ToF.FB_AoA_ToF_estimator, 1, False, True)
    }
    for method, (estimator, subc_stride, avg, temp_smoothed) in methods.items():
        if method in args.AoA_ToF_methods:
            if estimator == AoA_ToF.AoA_ToF_estimator:
                estimator(
                    CSI,
                    subc_stride=subc_stride, #4
                    stream_window_size=(args.num_Rx + 2) // 2, #256
                    subc_window_size=args.num_subcarriers // 2,
                    args=args,
                    avg=avg,
                    temp_smoothed=temp_smoothed,
                    ground_truth=ground_truth
                )
            elif estimator == AoA_ToF.FB_AoA_ToF_estimator:
                estimator(
                    CSI,
                    subc_stride=subc_stride, 
                    subc_window_size=args.num_subcarriers // 2,
                    args=args,
                    avg=avg,
                    temp_smoothed=temp_smoothed,
                    ground_truth=ground_truth
                )

class find_Peaks:
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
                stream_window_size, #3
                subc_window_size, #256
                subc_stride #4
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
        block_size = subc_window_size // subc_stride #256//4 =64(64*64)
        #32//4 = 8 (8*8: block size for each stream)
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


        #step1. Covariance matrix
        if len(x.shape) == 2:
            x = np.asarray(x, dtype=complex)
            S = x @ x.conj().T
        elif len(x.shape) == 3:
            S = 0
            for i in range(x.shape[0]):
                temp_x = np.asarray(x[i], dtype=complex)
                S += temp_x @ temp_x.conj().T
            S = S / x.shape[0]

        #step2. Eigen decomposition
        eig_val, eig_vec = np.linalg.eigh(S)
        eig_vec = eig_vec.astype(complex)
        idx_order = eig_val.argsort()[::-1]
        eig_val = eig_val[idx_order]
        eig_vec = eig_vec[:, idx_order]

        eig_val_db = 10 * np.log10(eig_val / np.max(eig_val))
        '''
        plt.figure()
        plt.plot(idx_order, eig_val_db, marker='o')
        plt.xlabel('Eigenvalue index (sorted)')
        plt.ylabel('Eigenvalue (dB, normalized)')
        plt.title('Eigenvalue Spectrum in dB')
        plt.grid(True)
        '''

        # Noise subspace
        Sdim = 11
        N_dim = eig_val.shape[0] - Sdim
        E_n = eig_vec[:, -N_dim:]
        P_n = E_n @ E_n.conj().T

        # theta candidate
        theta = np.arange(0, 360) if args.projection=='sin' else np.arange(0, 180)
        # tau candidate
        tau = np.arange(0.2*1e-8, 3.3*1e-8, 5e-10) #62 points

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
        row_size = subc_window_size // subc_stride #256//4=64
        sub_idx = np.arange(0, row_size) * subc_stride  #[0 4 8 ... 252] 64 points

        # carrier delay
        const_phase = np.exp(-2j * np.pi * fc * tau_j)
        # subcarrier frequency phase
        exp_omega = const_phase * np.exp(-2j * np.pi * delta_f * tau_j * sub_idx) #64 points

        # ----------------------
        # Spatial phase (AoA) #3
        # ----------------------
        if args.projection == "sin":
            exp_phi = np.exp(+2j * np.pi * fc * np.sin(theta_i) * args.d / 3e8 * np.arange(stream_window_size))
        elif args.projection == "cos":
            exp_phi = np.exp(2j * np.pi * fc * (1 - np.cos(theta_i)) * args.d / 3e8 * np.arange(stream_window_size))
        
        # ----------------------
        # Steering vector = Kronecker of spatial and subcarrier vectors
        # ----------------------
        steering_vector = np.kron(exp_phi, exp_omega) #3*64=192
        #print(f"Steering vector shape: {steering_vector.shape}")
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

