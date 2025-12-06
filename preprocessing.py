import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def show_preprocessing(csi, args):
    CSI_amp = np.abs(csi)
    CSI_phase = np.angle(csi)



    pha_methods = []
    pha_method_names = []

    if 'raw_without_offset' in args.sanitize_method:
        # (2.1) load raw CSI without phase offset for comparison
        data2 = np.load("CSI_remove_phase_offset.npz")
        #print(data.files) # "arr_0"
        X_CSIs = data2["arr_0"]
        CSI_phase_X = np.angle(X_CSIs)
        CSI_phase_X = np.unwrap(CSI_phase_X, axis = 0)
        CSI_phase_X = np.unwrap(CSI_phase_X, axis = -1)
        pha_methods.append(CSI_phase_X)
        pha_method_names.append('raw without offset')

    if 'raw' in args.sanitize_method:
        # (2.2) load raw CSI for comparison
        CSI_phase = np.unwrap(CSI_phase, axis = 0)
        CSI_phase = np.unwrap(CSI_phase, axis = -1)
        pha_methods.append(CSI_phase)
        pha_method_names.append('raw')

    if 'self_sanitize' in args.sanitize_method:
        CSI = Phase_sanitize.self_sanitize(csi, args)
        CSI = abs(CSI)
        CSI_phase = np.angle(CSI)
        pha_methods.append(CSI_phase)
        pha_method_names.append('self sanitize')

    if 'linear_fit' in args.sanitize_method:
        CSI_sanitize_lf = CSI_phase - Phase_sanitize.linear_fit(CSI_phase)
        pha_methods.append(CSI_sanitize_lf)
        pha_method_names.append('linear fit')

    if 'linear_regression' in args.sanitize_method:
        CSI_sanitize_lr  = CSI_phase - Phase_sanitize.linear_regression(CSI_phase)
        pha_methods.append(CSI_sanitize_lr)
        pha_method_names.append('linear reg')

    if 'sg_filter' in args.sanitize_method:
        csi_lr = Phase_sanitize.linear_regression(CSI_phase, real_time=True, window=49)
        csi_ts = Phase_sanitize.Time_Smoothing_sg(csi_lr, polyorder=2, real_time=True, window=49)
        CSI_sanitize_sg = CSI_phase - csi_ts
        pha_methods.append(CSI_sanitize_sg)
        pha_method_names.append('SG filter')

    if 'TSFR' in args.sanitize_method:
        csi_lr = CSI_phase - Phase_sanitize.linear_regression(CSI_phase, real_time=True, window=11)
        csi_ts_sg = Phase_sanitize.Time_Smoothing_sg(csi_lr, polyorder=2, real_time=True, window=11)
        CSI_sanitize_TSFR =  Phase_sanitize.Freq_Rebuild(csi_ts_sg, csi_lr, gamma = 2)
        pha_methods.append(CSI_sanitize_TSFR)
        pha_method_names.append('TSFR')

    # (4) show CSI_phase sanitize comparison
    Plot_csi.phase_sanitize_diff_subc(pha_methods, pha_method_names, args)
    Plot_csi.phase_sanitize_diff_time(pha_methods, pha_method_names, args, avg = False)

    # (5) show CSI_amp comparison along time
    amp_methods = []
    amp_method_names = []
    if 'raw_without_offset' in args.amp_process_method:
        # (2.1) load raw CSI without phase offset for comparison
        data2 = np.load("CSI_remove_phase_offset.npz")
        #print(data.files) # "arr_0"
        X_CSIs = data2["arr_0"]
        CSI_amp_X = np.abs(X_CSIs)
        CSI_amp_X -=Amp_sanitize.moving_average(CSI_amp_X, window_size = args.fs* 0.5)
        amp_methods.append(CSI_amp_X)
        amp_method_names.append('raw without offset -MA')

    if 'raw' in args.amp_process_method:
        amp_methods.append(CSI_amp)
        amp_method_names.append('raw')
    
    if 'moving_average' in args.amp_process_method:
        window_size = 0.5 * args.fs
        CSI_amp_MA = CSI_amp.copy()
        CSI_amp_MA = Amp_sanitize.moving_average(CSI_amp_MA, window_size)
        amp_methods.append(CSI_amp_MA)
        amp_method_names.append(f'moving average w={window_size}')

    if '2_moving_average' in args.amp_process_method:
        window_size1 = 0.5 * args.fs
        window_size2 = 0.2 * args.fs
        CSI_amp_2MA = CSI_amp.copy()
        CSI_amp_2MA -= Amp_sanitize.moving_average(CSI_amp_2MA, window_size1)
        CSI_amp_2MA -= Amp_sanitize.moving_average(CSI_amp_2MA, window_size2)
        amp_methods.append(CSI_amp_2MA)
        amp_method_names.append(f'moving average (w1={window_size1} w2={window_size2})')

    Plot_csi.amp_sanitize_diff_time(amp_methods, amp_method_names, args)




class Plot_csi:
    def rx_colors(i):
        colors = ['red', 'orange', 'green', 'royalblue', 'darkviolet', 'magenta', 'aqua', '']

        return colors[i % len(colors)]

    def colors(i):
        colors = ['gray', 'coral', 'royalblue', 'green', 'gold', 'purple', 'black', 'brown']
        return colors[i % len(colors)]

    def phase_sanitize_diff_subc(pha_methods, pha_method_names, args):
        # show phase sanitize comparison for Tx0->Rx0 @ args.plotted_packet
        subc = np.arange(args.num_subcarriers)  # 子載波軸
        t_idx, tx_idx, rx_idx = args.plotted_packet , 0, 0

        print(f"Phase Sanitize Comparison at time: {args.plotted_packet}, Tx{tx_idx} -> Rx{tx_idx}")
        print("Method Names:", pha_method_names)
        

        for i, method in enumerate(pha_methods):
            plt.figure(figsize=(5, 6), dpi=100)
            for rx_idx in range(args.num_Rx):   
                plt.plot(subc,
                        method[t_idx, tx_idx, rx_idx, :],
                        linewidth=1.5,
                        label=f"Rx{rx_idx}",
                        color=Plot_csi.rx_colors(rx_idx)
                        )
            plt.xlabel("Subcarrier Index")
            plt.ylabel("Phase (radians)")
            if pha_method_names[i] != 'raw' and pha_method_names[i] != 'self sanitize': plt.ylim(-np.pi, np.pi)
            plt.xlim(0, args.num_subcarriers - 1)
            plt.title(f"Phase vs subcarrier (@frame={t_idx}, Tx{tx_idx} → Rx {pha_method_names[i]})")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

    def phase_sanitize_diff_time(pha_methods, pha_method_names, args, avg = False):
        """
        顯示不同 phase sanitize 方法在時間軸上的相位變化
        固定: Tx0 → Rx0第0個subcarrier
        """
        time_axis = np.arange(args.time* args.fs)  # 時間軸 (T)
        tx_idx, rx_idx, subc_idx = 0, 0, 0          # 固定鏈路與子載波
        print(f"Phase Sanitize Comparison along time (subcarrier={subc_idx}), Tx{tx_idx} -> Rx{rx_idx}")
        print("Methods:", pha_method_names)


        for i, method in enumerate(pha_methods):
            plt.figure(figsize=(5, 6), dpi=100)
            for rx_idx in range(args.num_Rx):
                plt.plot(time_axis,
                        method[:, tx_idx, rx_idx, subc_idx],
                        linewidth=1.2,
                        label=f"Rx{rx_idx}",
                        color=Plot_csi.rx_colors(rx_idx))
            
            plt.title(f"Phase vs. Time (Tx{tx_idx} → Rx, Subcarrier {subc_idx}), {pha_method_names[i]}")
            plt.xlabel("Time Index (Frame)")
            plt.ylabel("Phase (radians)")
            plt.ylim(-5, 5)
            plt.legend(frameon=False)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()

    def amp_sanitize_diff_time(amp_methods, amp_method_names, args):
        time_axis = np.arange(args.time* args.fs)  # 時間軸 (T)
        plt.figure(figsize=(8, 5), dpi=100)
        tx_idx, rx_idx, subc_idx = 0, 0, 0
        for i, method in enumerate(amp_methods):
            plt.plot(time_axis,
                    method[:, 0, 0, 0],
                    linewidth=1.2,
                    label=amp_method_names[i],
                    color=Plot_csi.colors(i))
        plt.xlabel("Time Index (Frame)")
        plt.ylabel("Amplitude")
        plt.title(f"Amplitude Sanitization Comparison along time (Tx{tx_idx} → Rx{rx_idx}, Subcarrier {subc_idx})")
        plt.legend(frameon=False)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

class Phase_sanitize:

    def self_sanitize(x, args):
        mag = np.abs(x)
        mag[mag == 0] = 1
        return x * np.conj(x) / mag

    def linear_fit(csi_phase):
        T, num_Tx, num_Rx, num_subc = csi_phase.shape
        subc = np.arange(num_subc)
        linear_fit_phase = np.zeros_like(csi_phase)

        for t in range(T):
            for j in range(num_Tx):  
                for i in range(num_Rx):
                    phi = csi_phase[t, j, i, :]

                    # 斜率 (SFO) : (last_phase - first_phase) / (num_subc -1)
                    alpha = (phi[-1] - phi[0]) / (num_subc - 1)
                    # 平均值 (CFO) : mean(phi)
                    beta = np.mean(phi)
                    #print("alpha:", alpha)
                    #print("beta:", beta)

                    #linear fit
                    beta = 0
                    phi_fit = alpha * subc + beta

                    linear_fit_phase[t, j, i, :] = phi_fit

        return linear_fit_phase

    def linear_regression(csi_phase, real_time=True, window=49):
        """
        對 CSI 相位沿 subcarrier 方向做線性回歸去趨勢
        real_time=True 時使用滑動視窗平均
        """

        T, num_Tx, num_Rx, num_subc = csi_phase.shape
        subc = np.arange(num_subc)
        linear_reg_phase = np.zeros_like(csi_phase)

        if real_time:
            # --- Sliding window 平均版本 ---
            half_window = window // 2

            for t in range(T):
                # 計算視窗邊界
                start = max(0, t - half_window)
                end   = min(T, t + half_window + 1)

                # 平均該時間區段的相位
                phi_window = np.mean(csi_phase[start:end, :, :, :], axis=0)

                for j in range(num_Tx):
                    for i in range(num_Rx):
                        phi = phi_window[j, i, :]

                        # 線性擬合：phi ≈ α·subc + β
                        alpha, beta = np.polyfit(subc, phi, 1)
                        phi_fit = alpha * subc + beta

                        linear_reg_phase[t, j, i, :] = phi_fit

            print(f"[linear_regression: real_time] window={window}, shape={linear_reg_phase.shape}")
            return linear_reg_phase
        else:
            # --- 非即時版本 (逐幀擬合) ---
            for t in range(T):
                for j in range(num_Tx):
                    for i in range(num_Rx):
                        phi = csi_phase[t, j, i, :]
                        alpha, beta = np.polyfit(subc, phi, 1)

                        # 同樣用中心化版本，避免跨 Rx 偏移
                        subc_mean = np.mean(subc)
                        phi_mean = np.mean(phi)
                        phi_fit = alpha * (subc - subc_mean) + phi_mean

                        linear_reg_phase[t, j, i, :] = phi_fit

            print(f"[linear_regression: offline] shape={linear_reg_phase.shape}")
            return linear_reg_phase
    
    def Time_Smoothing_sg(csi_lr, polyorder=2, real_time=True, window=49):
        T, num_Tx, num_Rx, num_subc = csi_lr.shape
        window_length = window
        
        if real_time:
            csi_phase_sg = np.copy(csi_lr) 
            for tx in range(num_Tx):
                for rx in range(num_Rx):
                    for sc in range(num_subc):
                        csi_phase_sg[:, tx, rx, sc] = savgol_filter(
                            csi_lr[:, tx, rx, sc],
                            window_length, polyorder, mode='interp'
                        )
            print("csi_phase_sg shape (corr=1):", csi_phase_sg.shape)
            return csi_phase_sg  
        
    def Freq_Rebuild(csi_sg_phase, csi_lr_phase, gamma = 1, real_time=True, window=49):
        if real_time:
            T, num_Tx, num_Rx, num_subc = csi_lr_phase.shape
            window_half = window // 2
            csi_phase_corr = np.copy(csi_sg_phase)
            counter = 0

            # --- unwrap 沿子載波方向 ---
            csi_sg_phase = np.unwrap(csi_sg_phase, axis=-1)
            csi_lr_phase = np.unwrap(csi_lr_phase, axis=-1)

            for s in range(T):
                # Step 1️⃣：取時間滑動視窗內的資料
                start = max(0, s - window_half)
                end   = min(T, s + window_half + 1)

                # ⚠️ 改回使用 csi_lr_phase 計算差分（Eq.14）
                diff_window = np.diff(csi_lr_phase[start:end, :, :, :], axis=-1)  # shape: (W, Tx, Rx, Subc-1)

                # 平均與標準差以時間視窗內取平均（Eq.14）
                mu_s  = np.mean(diff_window, axis=0)   # (Tx, Rx, Subc-1)
                std_s = np.std(diff_window, axis=0)
                d_s   = mu_s + gamma * std_s

                # Step 2️⃣：沿子載波逐步修正 (Eq.17)
                for j in range(num_Tx):
                    for i in range(num_Rx):
                        for k in range(1, num_subc):
                            eps = csi_sg_phase[s, j, i, k] - csi_sg_phase[s, j, i, k-1]
                            thr = d_s[j, i, k-1]

                            if eps < -thr:
                                csi_phase_corr[s, j, i, k] = csi_phase_corr[s, j, i, k-1] - thr
                                counter += 1
                            elif eps > thr:
                                csi_phase_corr[s, j, i, k] = csi_phase_corr[s, j, i, k-1] + thr
                                counter += 1
                            else:
                                csi_phase_corr[s, j, i, k] = (
                                    csi_sg_phase[s, j, i, k]
                                    - (csi_sg_phase[s, j, i, k-1] - csi_phase_corr[s, j, i, k-1])
                                )

            #print(f"[TSFR sliding] gamma={gamma}, 修正點數={counter}, 平均每 frame={counter/(T*num_Rx):.3f}")
            return csi_phase_corr

        else:
            """
            Frequency Rebuild (TSFR Eq.(17))
            csi_phase_sg: SG 濾波後的相位 (T, Tx, Rx, Subc)
            csi_phase_lr: LR 校正後的相位 (T, Tx, Rx, Subc)
            """
            T, num_Tx, num_Rx, num_subc = csi_lr_phase.shape

            # unwrap 沿 subcarrier 方向
            csi_sg_phase = np.unwrap(csi_sg_phase, axis=-1)
            csi_lr_phase = np.unwrap(csi_lr_phase, axis=-1)

            # Step 3: threshold (Eq.14)
            # defined by csi_phase_lr
            diff = np.diff(csi_sg_phase, axis=-1)  # subcarrier-wise diff
            mu_s = np.mean(diff, axis=-1, keepdims=True)
            std_dev = np.std(diff, axis=-1, keepdims=True)

            # gamma 調整閾值(ds)
            d_s = mu_s + gamma * std_dev  # shape: (T, Tx, Rx, 1)

            csi_phase_corr = np.copy(csi_sg_phase)
            counter = 0
            # ---- 正確 loop ----
            for s in range(T):           
                for j in range(num_Tx):      
                    for i in range(num_Rx): 
                        for k in range(1, num_subc):  
                            eps = csi_sg_phase[s, j, i, k] - csi_sg_phase[s, j, i, k-1]
                            thr = d_s[s, j, i, 0]

                            if eps < -thr:
                                csi_phase_corr[s, j, i, k] = csi_phase_corr[s, j, i, k-1] - thr
                                counter += 1
                            elif eps > thr:
                                csi_phase_corr[s, j, i, k] = csi_phase_corr[s, j, i, k-1] + thr
                                counter += 1
                            else:
                                csi_phase_corr[s, j, i, k] = (
                                    csi_sg_phase[s, j, i, k]
                                    - (csi_sg_phase[s, j, i, k-1] - csi_phase_corr[s, j, i, k-1])
                                )
            #counter 計算有多少點被修正

            #print(f"gamma: {gamma}, Freq_Rebuild ratio: {counter/512.0}")
            return csi_phase_corr
 
   
class Amp_sanitize:

    def moving_average(csi_amp, window_size):
        """
        對每個 (Tx, stream, subcarrier) 在時間軸上做移動平均
        Input:
            data.shape = (num_frames, num_Tx, num_streams, num_subcarriers)
        Output:
            same shape
        """
        window_size = int(round(window_size))
        window = np.ones(window_size) / window_size
        
        return np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=0, arr=csi_amp)

        """
        Gain estimation for uniformly-spaced G (Eq.16–18)
        delta_db : ACG level等距增益間隔 (dB)
        Trep     : frame 間隔 (秒)
        """
        # Step 1: 每幀平均功率 (dB)
        Gamma = 10 * np.log10(np.mean(np.abs(csi_amp)**2, axis=-1))  # shape = (T, Tx, Rx)
        T, Tx, Rx = Gamma.shape
        g_hat = np.zeros((T, Tx, Rx))

        for tx in range(Tx):
            for rx in range(Rx):
                # Step 2: 時間平滑 (低通成分 g1)
                window = int(max(1, round(1/(Trep*10))))  # 約0.1 Hz cutoff
                g1 = np.convolve(Gamma[:, tx, rx],
                                np.ones(window)/window, mode='same')

                # Step 3: 離散化為等距格點 (ΔG)
                g2_continuous = Gamma[:, tx, rx] - g1
                g2_uniform = np.round(g2_continuous / delta_db) * delta_db

                # Step 4: 組合並轉回線性尺度
                g_hat[:, tx, rx] = 10 ** ((g1 + g2_uniform) / 20)

        g_hat = g_hat[..., None]
        return csi_amp / g_hat