#!/bin/bash

python main.py --mode train_with_pseudo_labels --checkpoint NREP_PS --logger ./logger/NREP_PS/ --noise_ratio_file noise_ratios.npz --epochs 10 --audio_dir /data/CLAP/features --video_dir /data/CLIP/features --lamda_1 1.0 --lamda_2 0.01 --lamda_3 1.0 --lamda_4 0.01
