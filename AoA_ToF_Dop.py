import numpy as np
import matplotlib.pyplot as plt
from AoA_ToF import*


def evaluate_AoA_Tof_Doppler_methods(args, CSI):
    subc_window_size=args.num_subcarriers // 2 #256
    #smoothed aoa-tof size = (sub_ws // sub_stride *stream_ws)^2 = (256//64 * 3)^2= (12x12) =144
    subc_stride = 32
    args.nperseg = 4 
    #smoothed aoa-tof-dop size = (sub_ws // sub_stride *stream_ws * nperseg)^2 = (144*3)^2 = (432x432)
    sampled_x = AoA_ToF_Doppler.sampled_AoA_ToF_Dop(
        CSI, 
        args, 
        stream_window_size =(args.num_Rx + 2) // 2,
        subc_window_size = subc_window_size, 
        subc_stride = subc_stride,
        nperseg = args.nperseg
        )
    #print("sampled_x.shape: ", sampled_x.shape)
    theta, tau, freqs, P_aoa_tof_w = AoA_ToF_Doppler.cal_AoA_ToF_Dop(
        sampled_x, 
        args, 
        stream_window_size =(args.num_Rx + 2) // 2,
        subc_window_size = subc_window_size, 
        subc_stride = subc_stride,
        nperseg = args.nperseg
        )
    
    plt.figure()
    plt.imshow(
        P_aoa_tof_w,
        extent=[tau.min(), tau.max(), theta.max(), theta.min()],
        aspect='auto',
        origin='upper'
    )
    plt.xlabel("Delay τ (s)")
    plt.ylabel("AoA θ (deg)")
    plt.title("AoA–ToF MUSIC (|f_D|-weighted, linear→dB)")
    plt.colorbar(label="dB")
    plt.show()
    pass

class AoA_ToF_Doppler:
    def plot_AoA_ToF_Doppler(theta, tau, freqs, P_music, title=""):
        # 例：先用 max 版本
        P_aoa_tof = P_music.max(axis=2)  # or 用 sum / weighted sum

        # 畫 AoA–ToF heatmap
        plt.figure()
        plt.imshow(
            P_aoa_tof,
            extent=[tau.min(), tau.max(), theta.max(), theta.min()],
            aspect='auto',
            origin='upper'
        )
        plt.xlabel("Delay τ (s)")
        plt.ylabel("AoA θ (deg)")
        plt.title("AoA-ToF MUSIC")
        plt.colorbar(label="dB")
        plt.show()

    def cal_AoA_ToF_Dop(x, args, stream_window_size, subc_window_size, subc_stride, nperseg):
        print("Calculating AoA-ToF-Doppler spectrum using MUSIC...")
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

        '''
        eig_val_db = 10 * np.log10(eig_val / np.max(eig_val))
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
        # doppler frequency candidate
        freqs = np.arange(-30, -20)
        # total points:  181 * 62 * 81 = 912402
        '''
        P_music = np.zeros([theta.shape[0], tau.shape[0], freqs.shape[0]])
        for k in range(len(freqs)):
            print(f"calculating freq {k+1}/{len(freqs)}")
            for i in range(len(theta)):
                for j in range(len(tau)):
                    # theta_i = i, tau_j = j, freq_k = k
                    steering_vector = AoA_ToF_Doppler.steering_vector_AoA_ToF_Dop(
                        theta[i], 
                        tau[j], 
                        freqs[k], 
                        args, 
                        stream_window_size, 
                        subc_window_size, 
                        subc_stride,
                        nperseg
                        )
                    PP = steering_vector.conj().T @ P_n @ steering_vector
                    #P_music[i, j, k] = 10 * np.log10(abs(1 / PP)+ 1e-12)
                    P_music[i, j, k] = (abs(1 / PP)+ 1e-12) #linear for weighting
        return theta, tau, freqs, P_music
        '''
        theta = np.arange(0, 360) if args.projection=='sin' else np.arange(0, 180)
        tau   = np.arange(0.2*1e-8, 3.3*1e-8, 5e-10)
        freqs = np.arange(-25, -20)

        n_theta = len(theta)
        n_tau   = len(tau)
        n_freqs = len(freqs)

        # ---- 建立 |fd| 權重 ----
        w = np.abs(freqs).astype(float)   # (n_freqs,)
        if np.all(w == 0):
            w = np.ones_like(w)
        w = w / (w.sum() + 1e-12)

        # ---- 初始化「加權後 AoA–ToF 能量」(線性刻度) ----
        P_aoa_tof_lin = np.zeros((n_theta, n_tau), dtype=float)

        # ---- 掃 freq，同時做 |fd| 加權 sum ----
        for k in range(n_freqs):
            print(f"calculating freq {k+1}/{n_freqs}")
            wk = w[k]   # 這個 fd 的權重

            for i in range(n_theta):
                for j in range(n_tau):
                    steering_vector = AoA_ToF_Doppler.steering_vector_AoA_ToF_Dop(
                        theta[i],
                        tau[j],
                        freqs[k],
                        args,
                        stream_window_size,
                        subc_window_size,
                        subc_stride,
                        nperseg
                    )
                    PP = steering_vector.conj().T @ P_n @ steering_vector
                    val = np.abs(1.0 / (PP + 1e-12))    # 線性 MUSIC 值
                    P_aoa_tof_lin[i, j] += wk * val     # 累加加權能量

        # ---- 線性 → dB ----
        P_aoa_tof_w = 10 * np.log10(P_aoa_tof_lin + 1e-12)

        return theta, tau, freqs, P_aoa_tof_w


    def sampled_AoA_ToF_Dop(CSI, args, stream_window_size,subc_window_size, subc_stride, nperseg):
        print("Sampling AoA-ToF-Doppler spectrum using MUSIC...")
        if args.plotted_packet + args.nperseg > args.num_frames:
            raise ValueError(
            f"Time window out of range: plotted_packet={args.plotted_packet}, "
            f"nperseg={args.nperseg}, num_frames={args.num_frames}"
            )
        #CSI.shape = (num_frames, num_Tx, num_Rx, num_subcarriers)
        #sample Rxs
        #sample subcarriers
        #sample frames
        #take smoothed AoA-ToF samples
        sampled_x = []
        for i in range(nperseg):
            csi = CSI[args.plotted_packet+ i]
            x = AoA_ToF.smoothed_csi(csi, args, stream_window_size, subc_window_size, subc_stride)#smooth fix tx_idx =0
            print(f"smoothed aoa-tof csi shape: {x.shape}")
            sampled_x.append(x)
        sampled_x = np.array(sampled_x) #(nperseg, Rx, subc)
        sampled_x = np.transpose(sampled_x, (1,2,0)) #(Rx, subc, nperseg)
        #reshape to Rx*subc*nperseg
        sampled_x = sampled_x.reshape(-1, 1)
        return sampled_x
    
    def steering_vector_AoA_ToF_Dop(theta_i, tau_j, freq_k, args, stream_window_size, subc_window_size, subc_stride, nperseg):

        fc = args.f_0
        delta_f = args.delta_f
        theta_i = np.deg2rad(theta_i)
        
        row_size = subc_window_size // subc_stride  # number of subcarriers after stride 32//8 =4
        sub_idx = np.arange(0, row_size) * subc_stride   # indices of selected subcarriers #[0 8 16 24] 4 points

        const_phase = np.exp(-2j * np.pi * fc * tau_j)
        exp_omega = const_phase * np.exp(-2j * np.pi * delta_f * tau_j * sub_idx) #4 points
        #print(f"stream window size: {stream_window_size}, subc window size: {subc_window_size}, subc stride: {subc_stride}")

        if args.projection == "sin": # 3
            exp_phi = np.exp(+2j * np.pi * fc * np.sin(theta_i) * args.d / 3e8 * np.arange(stream_window_size))
        elif args.projection == "cos":
            exp_phi = np.exp(2j * np.pi * fc * (1 - np.cos(theta_i)) * args.d / 3e8 * np.arange(stream_window_size))

        sv_AoA_ToF = np.kron(exp_phi, exp_omega) #3*4=12

        sv_AoA_ToF = sv_AoA_ToF.reshape(-1, 1)
        #print(f"steering_vector_AoA_ToF shape: {sv_AoA_ToF.shape}")
        sv_AoA_ToFs= []
        for i in range(len(sv_AoA_ToF)):
            sv_AoA_ToFs.append(sv_AoA_ToF)
        #print(f"sv_AoA_ToFs shape: {np.array(sv_AoA_ToFs).shape}")
            
        

        exp_omega = np.exp(1j * 2 * np.pi * np.arange(0, args.nperseg) * freq_k / args.fs)
        sv_AoA_ToF_Dop = np.kron(sv_AoA_ToFs, exp_omega)
        sv_AoA_ToF_Dop = sv_AoA_ToF_Dop.reshape(-1, 1)
        #print(f"steering_vector_AoA_ToF_Dop shape: {sv_AoA_ToF_Dop.shape}")

        return sv_AoA_ToF_Dop
        