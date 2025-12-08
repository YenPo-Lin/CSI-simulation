import numpy as np
import matplotlib.pyplot as plt

# AoA-Doppler settings:
subcarrier_id = 0
tx_id = 0
npersub = 3

def evaluate_AoA_Doppler_methods(args, CSI):

    if 'MUSIC' in args.AoA_Doppler_methods:
        input_CSI = np.zeros((args.num_Rx, args.nperseg), dtype=complex)  # 初始化輸出
        for i in range(args.num_Rx):
            # 提取每個接收天線（Rx）在指定子載波範圍內的 CSI 時間片段
            input_CSI[i, :] = CSI[args.plotted_packet:args.plotted_packet + args.nperseg, 0, i, 0]   
        sampled_CSI = input_CSI.reshape(-1, 1)
        f, theta, P_music = ref.cal_AoA_Doppler_MUSIC(sampled_CSI, args)
        
        peaks = find_Peaks.find_peak_aoa_doppler(P_music, theta, f)
        for peak in peaks:
            print(f"AoA: {peak['theta']}, f: {peak['freq']}, P_music: {peak['value']}")
        ref.plot_AoA_Doppler(f, theta, P_music, title = "(MUSIC)")

    if 'smoothed' in args.AoA_Doppler_methods:
        sampled_CSI = ref.sampled_CSI_AoA_Doppler(CSI, args, tx_id = tx_id, subcarrier_id = subcarrier_id)
        smoothed_CSI = ref.smooth_CSI_AoA_f_doppler(sampled_CSI, args)
        f, theta, P_music = ref.cal_AoA_Doppler_smoothed(smoothed_CSI, args)
        ref.plot_AoA_Doppler(f, theta, P_music, title = "(smoothed)")

    if 'smoothed_avg' in args.AoA_Doppler_methods:
        #npersub = 10
        sub_interval = args.num_subcarriers // npersub
        smoothed_CSIs = []
        for i in range(5):
            sampled_CSI = ref.sampled_CSI_AoA_Doppler(CSI, args, tx_id = tx_id, subcarrier_id = i)
            smoothed_CSI = ref.smooth_CSI_AoA_f_doppler(sampled_CSI, args)
            smoothed_CSIs.append(smoothed_CSI)
        smoothed_CSIs = np.array(smoothed_CSIs)
        print("smoothed_CSIs shape:", smoothed_CSIs.shape)
        f, theta, P_music = ref.cal_AoA_Doppler_smoothed(smoothed_CSIs, args)
        ref.plot_AoA_Doppler(f, theta, P_music, title = "(smoothed subc-avg)")

    if 'Beamform' in args.AoA_Doppler_methods: 

        sampled_CSI = ref.sampled_CSI_AoA_Doppler(CSI, args, tx_id = tx_id, subcarrier_id = subcarrier_id)   
        f, theta, P_music = ref.cal_AoA_Doppler_beamform(sampled_CSI, args=args)
        P_music = np.flipud(P_music)
        peaks = find_Peaks.find_peak_aoa_doppler_interp(P_music, theta = theta, freqs= f)
        for peak in peaks:
            print(f"AoA: {peak['theta']}, f: {peak['freq']}, P_music: {peak['value']}")
        ref.plot_AoA_Doppler(f, theta, P_music, title = "(beamform)")

    if 'Beamform_avg' in args.AoA_Doppler_methods:
        #npersub = 10
        sub_interval = args.num_subcarriers // npersub
        sampled_CSIs = []
        for i in range(3):
            sampled_CSI = ref.sampled_CSI_AoA_Doppler(CSI, args, tx_id = tx_id, subcarrier_id=i)
            sampled_CSIs.append(sampled_CSI)
        sampled_CSIs = np.array(sampled_CSIs)
        print("Beamform_sampled_CSIs.shape: ",sampled_CSIs.shape)

        f, theta, P_music = ref.cal_AoA_Doppler_beamform(sampled_CSIs, args=args)
        P_music = np.flipud(P_music)
        ref.plot_AoA_Doppler(f, theta, P_music, title = "(beamform subc-avg)")

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

    def find_peak_aoa_doppler(P_mvdr, theta, freqs, top_k=2, threshold_ratio=0.1):
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

class ref:
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
                steering_vector = ref.steering_vector_AoA_Doppler_MUSIC(theta[i], freqs[j], args)
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

        if args.projection == "sin":
            exp_phi = np.exp( -2j * np.pi * args.f_0 * np.sin(theta_i) * args.d / 3e8 * np.arange(args.num_Rx))
        elif args.projection == "cos":  
            exp_phi = np.exp(  2j * np.pi * args.f_0 * (1 - np.cos(theta_i)) * args.d / 3e8 * np.arange(args.num_Rx))
        #vector exp_phi (doppler term)
        #[ 1, e^jω, e^j2ω, e^j3ω, ...e^j(Nstream * Nsubc -1)φ ]
        exp_omega = np.exp(1j * 2 * np.pi * np.arange(0, args.nperseg) * f_j / args.fs)
        steering_vector = np.kron(exp_phi, exp_omega)
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
        freqs = np.arange(-40, 40, 1)
        
        # calculate P_music
        #P_music(freq, theta) = 1/a^H E_n E_n^H a
        P_music = np.zeros([theta.shape[0], freqs.shape[0]])
        for i in range(theta.shape[0]):
            for j in range(freqs.shape[0]):		
                steering_vector = ref.steering_vector_AoA_Doppler_smoothed(theta[i], freqs[j], args)
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
        freqs = np.arange(-40, 40)
        
        P_mvdr = np.zeros([theta.shape[0], freqs.shape[0]])

        for i in range(theta.shape[0]):
            steering_vector = ref.steering_vector_AoA_Doppler_beamform(theta[i], args)
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
