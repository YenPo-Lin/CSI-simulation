import argparse
import numpy as np
from signal_processing import*
import os
import time



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
	# room config
    parser.add_argument('--room_width', type = float, default = 3)
    parser.add_argument('--room_height', type = float, default = 3)
    parser.add_argument('--wall_permittivity', type = float, default = 5.24) # concrete: 5.24

    # Tx config
    parser.add_argument('--x_Tx', type = float, default = [1.2])# 0.2 
    parser.add_argument('--y_Tx', type = float, default = [0.1])# 1.5

    # Rx config
    # parser.add_argument('--x_Rx', nargs = "+", type = float, default = [1.5, 1.515, 1.53, 1.545, 1.56, 1.575, 1.59, 1.605]) 
    # parser.add_argument('--y_Rx', nargs = "+", type = float, default = [0.1,  0.1,   0.1,  0.1,   0.1,  0.1,  0.1,  0.1]) # [5.5, 5.5,   5.5,  5.5  ]
    parser.add_argument('--x_Rx', nargs = "+", type = float, default = [1.5, 1.515, 1.53, 1.545, 1.56, ]) # [4.5, 4.515, 4.53, 4.545]
    parser.add_argument('--y_Rx', nargs = "+", type = float, default = [0.1,  0.1,   0.1,  0.1,   0.1, ]) # [5.5, 5.5,   5.5,  5.5  ]

	# obstacles config
    parser.add_argument('--x_obs', nargs = "+", type = float, default = []) # [1.5, 0.5, 2.5]
    parser.add_argument('--y_obs', nargs = "+", type = float, default = []) # [1.5, 1, 1]  
    parser.add_argument('--obs_permittivity', type = float, default = 5.24) # concrete: 5.24

	# objects config
    parser.add_argument('--x_obj', nargs = "+", type = float, default = [0.5, 2.5]) # [0.5]
    parser.add_argument('--y_obj', nargs = "+", type = float, default = [2, 2]) # [1.5]  
    parser.add_argument('--v_obj', nargs = "+", type = float, default = [1, 0.7])# 
    parser.add_argument('--start_time', nargs = "+", type = float, default = [1, 1]) # 
    parser.add_argument('--break_points', nargs = "+", type = float, default = [[[0.5, 1], ], [[2.5, 3], ]]) #[[[1.5, 2.5], [1.5, 0.5], [1.5, 1.5]], ]  
    parser.add_argument('--obj_permittivity', type = float, default = 35.8) # skin: 35.8

	# CSI config
    parser.add_argument('--f_0', type = float, default = 5e9)
    parser.add_argument('--BW', type = float, default = 160e6)
    parser.add_argument('--delta_f', type = float, default = 312.5e3)
    parser.add_argument('--init_power', type = float, default = 10)
    parser.add_argument('--remove_reflection_material', dest = "remove_reflection_material", action = "store_true")
    parser.set_defaults(remove_reflection_material = False)
    parser.add_argument('--Tx_gain', type = float, default = 1)
    parser.add_argument('--Rx_gain', type = float, default = 1)
    parser.add_argument('--wall_path_num', type = int, default = 2)
    parser.add_argument('--obs_path_num', type = int, default = 2)
    parser.add_argument('--obj_path_num', type = int, default = 5)
    parser.add_argument('--reflective_ratio', type = float, default = 0.5)
    parser.add_argument('--length_ratio', type = float, default = 0.9)
    parser.add_argument('--remove_noise', dest = "remove_noise", action = "store_true")
    parser.set_defaults(remove_noise = False)
    parser.add_argument('--snr', type = float, default = 10)
    parser.add_argument('--remove_phase_offset', dest = "remove_phase_offset", action = "store_true")
    parser.set_defaults(remove_phase_offset = False)
    parser.add_argument('--CFO_PPM', type = float, default = 20)
    parser.add_argument('--OFDM_fs', type = float, default = 5e6)
    parser.add_argument('--SFO_PPM', type = float, default = 20)
    parser.add_argument('--PDD_max', type = float, default = 2.7e-7)
    parser.add_argument('--PDD_min', type = float, default = 2.3e-7)
    parser.add_argument('--PA_max', type = float, default = 200)
    parser.add_argument('--PA_min', type = float, default = 50)


    #global config
    parser.add_argument('--time', type = float, default = 3)
    parser.add_argument('--fs', type = float, default = 100)
    parser.add_argument('--plotted_packet', type = int, default = 120)
    parser.add_argument('--seed', type=int, default=777)

    #signal processing config
    parser.add_argument('--sanitize_method', nargs = "+", default = ['raw_without_offset', 'raw', 'TSFR', '','', '', '', ''])
    parser.add_argument('--amp_process_method', nargs = "+", default = ['', '', '', '','', '', '', ''])
    
    parser.add_argument('--mD_estimation', nargs = "+", default = ['', 'ref', '', '', '', ''])

    parser.add_argument('--AoA_ToF_methods', nargs = "+", default = ['FB_smoothed', 'smoothed_avg', 'smoothed', '', '', '', '', '', '', '', '', '', '']) #['smoothed_avg', 'smoothed', '', '']
    parser.add_argument('--AoA_Doppler_methods', type = str, default = ['', '', '', '', '', '', '']) #['smoothed_avg', 'smoothed', '', '', '','']) 
    parser.add_argument('--projection', default = 'cos')
    #args
    args = parser.parse_args()
    # set seed
    np.random.seed(args.seed)

    #Time series
    T = np.arange(0, args.time, 1/args.fs)
    args.T = T

    #Antenna settings
    args.num_Tx = len(args.x_Tx)
    args.num_Rx = len(args.x_Rx)
    args.d = ((args.x_Rx[0] - args.x_Rx[1]) ** 2 + (args.y_Rx[0] - args.y_Rx[1]) ** 2) ** 0.5
    args.num_frames = len(T)
    args.num_subcarriers = int(args.BW/args.delta_f)
    args.freqs = np.array([args.delta_f * i for i in range(args.num_subcarriers)])
    args.nperseg = int(0.49 * args.fs)
    args.noverlap = args.nperseg - 1
    args.sub_nperseg = args.nperseg - 20

    

	# STFT setting
    STFT_setting = {
		'nperseg': int(0.49 * args.fs),
		'return_features': 57*2,
		'focus_freq': 40,
	}
    STFT_setting['noverlap'] =  STFT_setting['nperseg'] -1
    STFT_setting['nfft'] = int(STFT_setting['return_features'] / (2 * STFT_setting['focus_freq'] + 1) * args.fs)
    #print("STFT settings --- nfft: {}, nperseg: {}, noverlap: {}, return_features: {}".format(STFT_setting['nfft'], STFT_setting['nperseg'], STFT_setting['noverlap'], STFT_setting['return_features']))

    # information
    """
    print("===== WiFi Settings =====")
    for k in ['num_Tx', 'num_Rx', 'num_frames', 'num_subcarriers']:
        print(f"{k:<18}: {getattr(args, k)}")

    print("\n===== STFT Settings =====")
    for k in ['nfft', 'nperseg', 'noverlap', 'return_features']:
        print(f"{k:<18}: {STFT_setting[k]}")
    """



    # load
    datapath = "CSI_original.npz"
    #datapath = "CSI_remove_phase_offset.npz"
    data = np.load(datapath)
    raw_CSIs = data["arr_0"]
    x = raw_CSIs.copy()

    print("\n===== Input Data =====")
    print(f"{'CSI shape':<18}: {raw_CSIs.shape}")

    start_time = time.time()

    for i in range(1):
        signal_processing(x, args)

    end_time = time.time()

    print(f"{'Time':<18}: {end_time - start_time}")

    
    plt.show()
