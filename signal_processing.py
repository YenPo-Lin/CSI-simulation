import numpy as np
from signal_processing_utils import Reference as ref
from signal_processing_utils import AoA_ToF
from signal_processing_utils import find_Peaks as find_Peaks
from preprocessing import*
from AoA_ToF import evaluate_AoA_ToF_methods



def signal_processing(CSI, args):

    #show_preprocessing(CSI, args)

    print("Ref")
    CSI = Phase_sanitize.self_sanitize(CSI, args)
    CSI_phase = np.angle(CSI)
    CSI_amp = np.abs(CSI)
    CSI_amp -=Amp_sanitize.moving_average(CSI_amp, window_size = args.fs* 0.25)
    CSI = CSI_amp * np.exp(-1j * CSI_phase)
    

    #✅ joint estmation
    ground_truth = {
        "obj0": (58.79195929697761, 1.2754158439024172e-08),
        "obj1": (115.42867011533676, 1.5593202334168465e-08)
    }

    evaluate_AoA_ToF_methods(args, CSI, ground_truth)
    
    # AoA-Doppler settings:
    subcarrier_id = 0
    tx_id = 0
    npersub = 30

    if 'MUSIC' in args.AoA_Doppler_methods:
        input_CSI = np.zeros((args.num_Rx, args.nperseg), dtype=complex)  # 初始化輸出
        for i in range(args.num_Rx):
            # 提取每個接收天線（Rx）在指定子載波範圍內的 CSI 時間片段
            input_CSI[i, :] = CSI[args.plotted_packet:args.plotted_packet + args.nperseg, 0, i, 0]   
        sampled_CSI = input_CSI.reshape(-1, 1)
        f, theta, P_music = ref.cal_AoA_Doppler_MUSIC(sampled_CSI, args)
        peaks = find_Peaks.find_peak_aoa_doppler_v2(P_music, theta, f)
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

