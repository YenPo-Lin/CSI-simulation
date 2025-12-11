import numpy as np
from preprocessing import*
from AoA_ToF import evaluate_AoA_ToF_methods
from AoA_Dop import evaluate_AoA_Doppler_methods
from AoA_ToF_Dop import evaluate_AoA_Tof_Doppler_methods



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

    if 1: evaluate_AoA_ToF_methods(args, CSI, ground_truth)
    
    # AoA-Doppler settings:
    if 0: evaluate_AoA_Doppler_methods(args, CSI)
    
    if 0:evaluate_AoA_Tof_Doppler_methods(args, CSI)
    
