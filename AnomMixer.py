import sys
import glob
import os
import time

import random
import numpy as np
import wavio
import librosa
import matplotlib.pyplot as plt


# RY module import
lib_path = '/work-hmcomm/project/nedo2020_yokogawa/Git/yokogawa_ryamaguchi/nedo2020-yokogawass/script/RY_module/'
sys.path.insert(0,lib_path)

from utils.util import load_wav, humanize_time, my_makedirs


def outSpec(x,win_length,hop_length,normalize=False):
    '''
    スペクトラルを計算
    
    Parameters
    ----------
    x (numpy.array) : signal (monoral)
    win_length (int) : stft window size
    hop_length (int) : stft window stride
    normalize (bool) : 
    
    Returns
    -------
    D : spec
    '''
    x = x.squeeze()
    if normalize:
        x = np.double(x)/np.mean(x)
    # Save as complex valued output for mixing later
    D = librosa.stft(x.squeeze(),n_fft=win_length,hop_length=hop_length)
    return D

# Input must be raw STFT (complex values)
def calcPower(D):
    '''
    スペクトルパワーを計算し、それを周波数方向に積分し、そのmedian値を返す。
    
    Parameters
    ----------
    D : スペクトラル
    
    Returns
    -------
    パワーのメジアン値
    
    '''
    return np.median(20 * np.log10(np.sum(np.abs(D), axis=0)))


# 異音を調整する係数を算出する
def calcScaler(D,D_a,anr):
    P = calcPower(D)
    P_a = calcPower(D_a)
    alpha = anr - (P_a-P)
    return 10**(alpha/20)


def shape_anom_arr(anom_arr, sr):
    """
    異常音を1sより長いときに0.3sにする
    """
    anom_ = anom_arr.squeeze()
    if len(anom_) < sr:
        start_idx_anom = sr//2 - len(anom_)//2
        end_idx_anom = start_idx_anom + len(anom_)
        arr_vacant = np.zeros(sr)

        arr_vacant[start_idx_anom:end_idx_anom] = anom_
        anom_shape = anom_
    else:
#         random.seed(11)
#         start_idx = random.randint(0, len(anom_)-sr)
        anom_shape = anom_[sr:sr+int(sr*0.3)]
        
    return anom_shape


def wave_mixing(norm,anom,anr, is_oneshot=False):
    norm_=norm.squeeze()
    anom_=anom.squeeze()
    
    
    D_norm = outSpec(norm_, win_length=512, hop_length=256)
    D_anom = outSpec(anom_, win_length=512, hop_length=256)
    
    scaler = calcScaler(D_norm, D_anom, anr=anr)

    if is_oneshot:
        anom_ = np.concatenate([np.zeros(norm.shape[0]//2),
                                anom,
                                np.zeros(norm.shape[0] - norm.shape[0]//2 - anom.shape[0])]
                               ,axis=0)
    else:
        while anom_.shape[0] < norm_.shape[0]:
            anom_ = np.concatenate((anom_,anom_),axis=0)
        anom_ = anom_[:norm_.shape[0]] 
    
    return scaler*anom_+norm_

def arr_cut_random(arr, sr, cut_size, n):
    arr_list = []
    for i in range(n):
#         random.seed(11)
        cut_size = int(10*sr)
        start_idx = random.randint(0, len(arr)-cut_size)
        arr_shape = arr[start_idx:start_idx+cut_size]
        
        arr_list.append(arr_shape)
    return arr_list

def sample2ms(signal, sr):
    """
    sample数からmsに変換するヘルパー関数
    """
    n_sample = signal.shape[0]
    s = round(n_sample/sr, 3)
    ms = round(s * 1000)
    
    return ms

def test_sample_data():
    """
    ランダムに切り取った正常データがあるときに使う
    """
    # sample異常音を追加するためにしようしたparameter
    norm_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_mixing/norm/*_5_*cut_00*'
    anom_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/misc/PseudoAnomalyMixing/Anomaly_Mix_Samples/sample_sound_temp/*wav'
    output_dir = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/misc/PseudoAnomalyMixing/Anomaly_Mix_Samples_wavio-param_scale_is_none/'
    sr = 96000
    hop_length = int(sr * 0.2)
    win_length = int(sr * 0.4)
    
    is_oneshot = True
    is_shape100ms = True
    ####################
    
    # get path
    norm_path_list = sorted(glob.glob(norm_path))
    anom_path_list = sorted(glob.glob(anom_path))
    
    # load anom wav
    anom_arr_list = [load_wav(path, sr, is_mono=True) for path in anom_path_list]
    
    # sample異常音をついかするために使用
    anr_list = [-25, -20, -15]


    #################
    # main loop
    #################
    path_norm = f'{output_dir}norm/'
    path_anom = f'{output_dir}anom/'
    path_anom_oneshot = f'{output_dir}anom_oneshot/'
    my_makedirs(path_norm)
    my_makedirs(path_anom)
    my_makedirs(path_anom_oneshot)

    for norm_path in norm_path_list:
        print(norm_path)
        wav_name = os.path.splitext(os.path.basename(norm_path))[0]
        norm_name_split = wav_name.split('_')
        norm_name = norm_name_split[0] + '_'  + norm_name_split[1] + '_'  + norm_name_split[2]
        exp_id = norm_name_split[3]
        direc = norm_name_split[4]
        cut_id = norm_name_split[-1]

        # load norm wav
        norm_arr = load_wav(norm_path, sr, is_mono=True)

        # scale調整
        norm_arr = 2 ** 23 * 0.05 * (norm_arr/np.abs(norm_arr).mean())
        
        outputfile_name_norm = f'{path_norm}{wav_name}.wav'
        # wav file の書き込み
        if not os.path.exists(outputfile_name_norm):
            print(f'write to {outputfile_name_norm}')
            wavio.write(outputfile_name_norm, norm_arr, sr, sampwidth=3, scale='none')
        else:
            print(f'{outputfile_name_norm} is exists!')

        for anr in anr_list:
            print(f'start anr {anr}')

            for j, anom in enumerate(anom_arr_list):
                
                anom_name = os.path.splitext(os.path.basename(anom_path_list[j]))[0]
                print(f'mixing {anom_name}')
                
                if is_oneshot:
                    anom_ = shape_anom_arr(anom, sr)
                    
                    start_anom_time = sample2ms(norm_arr[:norm_arr.shape[0]//2], sr)
                    end_anom_time = sample2ms(anom_, sr)
                    outputfile_name_anom = f'{path_anom_oneshot}{norm_name}_{exp_id}_{anom_name}_{direc}_{anr:03}_{cut_id}_{start_anom_time}_{end_anom_time}.wav'

                    # wav file の書き込み
                    if not os.path.exists(outputfile_name_anom):
                        mix_arr = wave_mixing(norm_arr, anom_, anr, is_oneshot=True)
                        print(f'write to {outputfile_name_anom}')
                        wavio.write(outputfile_name_anom, mix_arr, sr, sampwidth=3, scale='none')
                    else:
                        print(f'{outputfile_name_anom} is exists!')
                
                # 異常サンプル音を0.1sだけ抜き取る
                if is_shape100ms:
                    anom = anom[:int(sr*0.1)]
                
                outputfile_name_anom = f'{path_anom}{norm_name}_{exp_id}_{anom_name}_{direc}_{anr:03}_{cut_id}.wav'

                if not os.path.exists(outputfile_name_anom):
                # wav file の書き込み
                    mix_arr = wave_mixing(norm_arr, anom, anr, is_oneshot=False)
                    print(f'write to {outputfile_name_anom}')
                    wavio.write(outputfile_name_anom, mix_arr, sr, sampwidth=3, scale='none')
                else:
                    print(f'{outputfile_name_anom} is exists!')
        print('*'*30)
        # break
def main_new(norm_path='/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_mixing/norm/*',
             anom_path='/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/misc/PseudoAnomalyMixing/Anomaly_Samples/anom*',
             output_dir='/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_mixing_corrected/',
             sr=96000,
             win_size=0.4,
             win_step=0.2,
             anr_list=[-30, -25, -20, -15],
             is_shape100ms=True,
             is_oneshot=True,
             is_test=False):
    """
    ランダムに切り取った正常データがあるときに使う
    """
    
    #  Parameter  #####
    # norm_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_mixing/norm/*'
    # anom_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/misc/PseudoAnomalyMixing/Anomaly_Samples/anom*'
    # output_dir = './output_mix_wav/'
    # output_dir = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_mixing/'
    # sr = 96000
    hop_length = int(sr * win_step)
    win_length = int(sr * win_size)
    
    # is_oneshot = True
    # is_shape100ms = True
    
    if is_test:
    # sample異常音を追加するためにしようしたparameter
        norm_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_mixing/norm/*_5_*cut_00*'
        anom_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/misc/PseudoAnomalyMixing/Anomaly_Mix_Samples/sample_sound_temp/*wav'
        output_dir = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/misc/PseudoAnomalyMixing/Anomaly_Mix_Samples/'
        sr = 96000
        hop_length = int(sr * 0.2)
        win_length = int(sr * 0.4)
        is_oneshot = True
        is_shape100ms = True
    ####################
    
    # get path
    norm_path_list = sorted(glob.glob(norm_path))
    anom_path_list = sorted(glob.glob(anom_path))
    
    # load anom wav
    anom_arr_list = [load_wav(path, sr, is_mono=True) for path in anom_path_list]

    #################
    # main loop
    #################
    path_norm = f'{output_dir}norm/'
    path_anom = f'{output_dir}anom/'
    path_anom_oneshot = f'{output_dir}anom_oneshot/'
    my_makedirs(path_norm)
    my_makedirs(path_anom)
    my_makedirs(path_anom_oneshot)

    for norm_path in norm_path_list:
        print(norm_path)
        wav_name = os.path.splitext(os.path.basename(norm_path))[0]
        norm_name_split = wav_name.split('_')
        norm_name = norm_name_split[0] + '_'  + norm_name_split[1] + '_'  + norm_name_split[2]
        exp_id = norm_name_split[3]
        direc = norm_name_split[4]
        cut_id = norm_name_split[-1]

        # load norm wav
        norm_arr = load_wav(norm_path, sr, is_mono=True)
        # norm data の scale 調整
        norm_arr = 2 ** 23 * 0.05 * (norm_arr / np.abs(norm_arr).mean())
        
        # norm wav file の書き込み
        outputfile_name_norm = f'{path_norm}{wav_name}.wav'
        if not os.path.exists(outputfile_name_norm):
            print(f'write to {outputfile_name_norm}')
            wavio.write(outputfile_name_norm, norm_arr, sr, sampwidth=3)
        else:
            print(f'{outputfile_name_norm} is exists!')


        for anr in anr_list:
            print(f'start anr {anr}')

            for j, anom in enumerate(anom_arr_list):
                
                anom_name = os.path.splitext(os.path.basename(anom_path_list[j]))[0]
                print(f'mixing {anom_name}')
                
                if is_oneshot:
                    anom_ = shape_anom_arr(anom, sr)
                    
                    start_anom_time = sample2ms(norm_arr[:norm_arr.shape[0]//2], sr)
                    end_anom_time = sample2ms(anom_, sr)
                   
                    outputfile_name_anom = f'{path_anom_oneshot}{norm_name}_{exp_id}_{anom_name}_{direc}_{anr:03}_{cut_id}_{start_anom_time}_{end_anom_time}.wav'

                    # anom oneshot wav file の書き込み
                    if not os.path.exists(outputfile_name_anom):
                        mix_arr = wave_mixing(norm_arr, anom_, anr, is_oneshot=True)
                        print(f'write to {outputfile_name_anom}')
                        wavio.write(outputfile_name_anom, mix_arr, sr, sampwidth=3)
                    else:
                        print(f'{outputfile_name_anom} is exists!')
                
                # 異常サンプル音を0.1sだけ抜き取る
                if is_shape100ms:
                    anom = anom[:int(sr*0.1)]
                
                outputfile_name_anom = f'{path_anom}{norm_name}_{exp_id}_{anom_name}_{direc}_{anr:03}_{cut_id}.wav'

                if not os.path.exists(outputfile_name_anom):
                # anom wav file の書き込み
                    mix_arr = wave_mixing(norm_arr, anom, anr, is_oneshot=False)
                    print(f'write to {outputfile_name_anom}')
                    wavio.write(outputfile_name_anom, mix_arr, sr, sampwidth=3)
                else:
                    print(f'{outputfile_name_anom} is exists!')
        print('*'*30)
    
def main_series():
    
    #  Parameter  #####
    norm_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_mixing/norm/*'
    anom_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/misc/PseudoAnomalyMixing/Anomaly_Samples/anom*'
    output_dir = './output_mix_wav/'
#     output_dir = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_mixing_corrected/'
    sr = 96000
    hop_length = int(sr * 0.2)
    win_length = int(sr * 0.4)
    norm_wav_cut_size = 10
    N_norm_wav_cut = 50
    ###################
    
    # get path
    norm_path_list = sorted(glob.glob(norm_path))
    anom_path_list = sorted(glob.glob(anom_path))
    
    # load anom wav
    anom_arr_list = [load_wav(path, sr, is_mono=True) for path in anom_path_list]
    # set anr list
    anr_list = [-30, -25, -20, -15, -10, -5, 0]
    
    #################
    # main loop
    #################
    path_anom = f'{output_dir}anom_oneshot/'
    my_makedirs(path_anom)

    for norm_path in norm_path_list:
        print(norm_path)
        wav_name = os.path.splitext(os.path.basename(norm_path))[0]
        norm_name_split = wav_name.split('_')
        norm_name = norm_name_split[0] + '_'  + norm_name_split[1] + '_'  + norm_name_split[2]
        exp_id = norm_name_split[3]
        direc = norm_name_split[4]
        cut_id = norm_name_split[-1]

        # load norm wav
        norm_arr = load_wav(norm_path, sr, is_mono=True)

        for anr in anr_list:
            print(f'start anr {anr}')

            for j, anom in enumerate(anom_arr_list):
                
                anom_name = os.path.splitext(os.path.basename(anom_path_list[j]))[0]
                print(f'mixing {anom_name}')
                
#                 anom_ = shape_anom_arr(anom, sr)
                
                mix_arr = wave_mixing(norm_arr, anom, anr, is_oneshot=False)

                start_anom_time = sample2ms(norm_arr[:norm_arr.shape[0]//2], sr)
                end_anom_time = sample2ms(anom_, sr)
                outputfile_name_anom = f'{path_anom}{norm_name}_{exp_id}_{anom_name}_{direc}_{anr:03}_{cut_id}_{start_anom_time}_{end_anom_time}.wav'

                # wav file の書き込み
                if not os.path.exists(outputfile_name_anom):
                    print(outputfile_name_anom)
                    wavio.write(outputfile_name_anom, mix_arr, sr, sampwidth=3)
                else:
                    print(f'{outputfile_name_anom} is exists!')
        print('*'*30)
        break

        
def main_old():
    #  Parameter  #####
    norm_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_cut/*wav'
    anom_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/misc/PseudAnomalyMixing/Anomaly_Samples/anom*'
    output_dir = './output_mix_wav/'
#     output_dir = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/wav_mixing/'
    sr = 96000
    hop_length = int(sr * 0.2)
    win_length = int(sr * 0.4)
    norm_wav_cut_size = 10
    N_norm_wav_cut = 50
    ###################   
    norm_path_list = sorted(glob.glob(norm_path))

    # 学習データ Exp06, Ex11　と　location 7 を除外する
    norm_path_list_ = [i for i in norm_path_list if (not 'Exp06' in i) and (not 'Exp11' in i) and (not '_7_' in i)]

    anom_path_list = sorted(glob.glob(anom_path))

    # wav load 
    anom_arr_list = [load_wav(path, sr, is_standard=True, is_mono=True) for path in anom_path_list]
    anr_list = [-30, -25, -20, -15, -10, -5, 0]
    
    ###############
    ## main loop ##
    ###############
    path_norm = f'{output_dir}norm/'
    my_makedirs(path_norm)
    path_anom = f'{output_dir}anom/'
    my_makedirs(path_anom)
    
    for norm_path in norm_path_list_:
        norm_name = os.path.splitext(os.path.basename(norm_path))[0]

        start_time = time.time()
        print(f'load {norm_name}')
        # wav load 
        norm_arr_left, norm_arr_right = load_wav(norm_path, sr)

        # wav cut norm_wav_cut_size[s] x N_norm_wav_cut
        norm_cut_left_list  = arr_cut_random(norm_arr_left, sr, norm_wav_cut_size, N_norm_wav_cut)
        norm_cut_right_list = arr_cut_random(norm_arr_right, sr, norm_wav_cut_size, N_norm_wav_cut)
 



        print('*'*30)

        print('start left ...')
        for i, norm_cut_left in enumerate(norm_cut_left_list):
            print(f'{i} / {len(norm_cut_left_list)}')

            outputfile_name_norm = f'{path_norm}{norm_name}_left_cut_{i:02}.wav'
            # wav file の書き込み
            if not os.path.exists(outputfile_name_norm):
                print(outputfile_name_norm)
                wavio.write(outputfile_name_norm, norm_cut_left, sr, sampwidth=3) 
            else:
                print(f'{outputfile_name_anom} is exists!')
            
            for anr in anr_list:
                print(f'start anr {anr}')
   
                for j, anom in enumerate(anom_arr_list):
                        anom_name = os.path.splitext(os.path.basename(anom_path_list[j]))[0]
                        print(f'mixing {anom_name}')

        #                 anom = shape_anom_arr(anom, sr)

                        mix_arr = wave_mixing(norm_cut_left, anom, anr)

                        outputfile_name_anom = f'{path_anom}{norm_name}_{anom_name}_left_{anr:03}_{i:02}.wav'

                        if not os.path.exists(path):
                            # wav file の書き込み
                            wavido.write(outputfile_name_anom, mix_arr, sr, sampwidth=3)
                            # wav file の書き込み
                            if not os.path.exists(outputfile_name_anom):
                                print(outputfile_name_anom)
                                wavio.write(outputfile_name_anom, mix_arr, sr, sampwidth=3)
                            else:
                                print(f'{outputfile_name_anom} is exists!')

        print('end left ...\n')

        print('start right ...')        
        for i, norm_cut_right in enumerate(norm_cut_right_list):
            print(f'{i} / {len(norm_cut_right_list)}')

            outputfile_name_norm = f'{path_norm}{norm_name}_right_cut_{i:02}.wav'
            # wav file の書き込み
            if not os.path.exists(outputfile_name_norm):
                print(outputfile_name_norm)
                wavio.write(outputfile_name_norm, norm_cut_left, sr, sampwidth=3)
            else:
                print(f'{outputfile_name_anom} is exists!') 
        
            for anr in anr_list:
                print(f'start anr {anr}')

                for j, anom in enumerate(anom_arr_list):
                    anom_name = os.path.splitext(os.path.basename(anom_path_list[j]))[0]
                    print(f'mixing {anom_name}')

#                     anom = shape_anom_arr(anom, sr)

                    mix_arr = wave_mixing(norm_cut_right, anom, anr)
    
                    outputfile_name_anom = f'{path_anom}{norm_name}_{anom_name}_right_{anr:03}_{i:02}.wav'

                    # wav file の書き込み
                    if not os.path.exists(outputfile_name_anom):
                        print(outputfile_name_anom)
                        wavio.write(outputfile_name_anom, mix_arr, sr, sampwidth=3)
                    else:
                        print(f'{outputfile_name_anom} is exists!')
        print('end right ...\n')
        print('*'*30)

#             break
        print(f'mixing is complete')
        print(f'total time : {humanize_time(time.time() - start_time )}')
        print(f'\n\n')
        break


if __name__ == '__main__':
    main_new()
    # test_sample_data()
 





