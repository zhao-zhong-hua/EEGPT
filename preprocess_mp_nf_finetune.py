
"""
Consider that there are n datasets (e.g. SEED, FACED, ..., (EEG+fNRIS), ...)
Each dataset has a partial of all possible channels
(e.g. SEED has only EEG chanels while some data has only fNRIS channels)
Each dataset is organized as [sub1.pkl, sub2.pkl, ...],
where each pkl is numpy array of shape[c, n_second * sample_rate]
(n_second and sample_rate can also be different across datasets),
this code would
    1. preprocess each dataset:
        1.1 slice data into windows (in unit of seconds, e.g. 10s / 5s)
        1.2 extract features (e.g. DE),
        1.3 and then save features in another pkl file
    2. build a pytorch Dataset that reads features randomly, and pad each datum to
        [max_n_channels, max_window_length, n_features],
        then return padded sample, and the pad mask (chan_pad) in __getitem__
"""
import time
import logging
from logger import setup_logger
import psutil
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import mne
import os

import ast
from multiprocessing import Pool
from functools import partial
from scipy import signal
from mne.io import RawArray
from eeg_channels import eeg_locs

#预处理先切片保存，10s左右一个，每次前进五s有重叠。
# 提取de特征时按照1s提取。
#get_item不能用属性维护，初始的时候要指定好idex对应的样本。
# getitem返回的时候，dataloder的时候使用collate_fn
#返回position(返回通道名)
#每个batch batchsize*chan*window*feature


#近红外频率太低。或者简单做下采样插值，把每秒17个值降为5个值。或者就用17个点。


# 一个预训练数据应该有的类型

class Subject_data():

    def __init__(self):
        self.subject_total_id = None  # 被试在所有数据集里的编号
        self.subject_gender = None  # 0 for female, 1 for male
        self.subject_age = None  # the age of the subject
        self.subject_dataset = None  # 数据集名称
        self.subject_id_dataset = None  # 这个被试在这个数据集里的编号
        self.subject_file_name = None  # 这个被处理的数据文件的原始名称
        self.subject_dataset_note = 'This is a dataset for LEM'  # 用于概括数据集的作用
        self.subject_label = None  # 一些关于被试的标记
        self.file_format = None
        self.hh = None

        self.eeg_info = 'no'  # 有EEG数据
        self.fnirs_info = 'no'  # 有近红外数据
        self.fmri_info = 'no'  # 用于未来数据集
        self.mri_info = 'no'  # 用于未来数据集

        self.eeg = None  # 具体的数据信息
        self.fnirs = None
        self.fmri = None
        self.mri = None
        self.preprocess = None
        self.add_preprocess_history()  # 初始化参数

    def add_eeg_data(self, device, channels, data, note):
        self.eeg = self.EEG_data(device, channels, data, note)
        self.eeg_info = 'yes'

    def add_fnirs_data(self, device, channels, data, note):
        self.fnirs = self.FNIRS_data(device, channels, data, note)
        self.fnirs_info = 'yes'

    def add_preprocess_history(self):  # 这一条全靠变量赋予
        self.preprocess = self.Preprocess_info()

    class EEG_data:
        def __init__(self, device, channels, data, note,eeg_locs):
            self.eeg_device = device
            self.eeg_channels = channels  # 这个被试真实有效的通道，不要参考通道
            self.eeg_chn_locs = eeg_locs  # 128个通道的位置信息  {'Fz':(1,1),'Oz':(100,12)}
            self.eeg_data = data  # 128个通道的信息  {'Fz':array, 'Oz':array, 'Cz':None, ..., 'Trigger': array}
            self.trigger_note = note  # trigger含义 '0 for positive video, 1 for neutral video , 2 for negative'

            # 后添加的
            self.fs = 250  # 采样率(250Hz)
            # self.freqs = [[1, 4], [4, 8], [8, 14], [14, 30], [30, 47]]  # 五个频率段
            self.freqs = [[1,3],[3,5],[5,8],[8,10],[10,12],[12,16],[16,20],[20,30],[30,40],[40,47]]
            # self.window_size = 4  # 单个切片的窗口大小(10秒) #这个参数不起作用，
            self.de_window_size = 1  # 提取DE特征的窗口大小(1秒)
            self.step = 3  # 切片窗口移动的步长(5秒)

        # def filter(self, data):
        #     n_freqs = len(self.freqs)  # 获取频率段数量
        #
        #     filtered_data = np.zeros((n_freqs, *data.shape))
        #
        #     # 遍历每个频率段
        #     for freq_idx, freq_range in enumerate(self.freqs):
        #         low_freq, high_freq = freq_range  # 获取当前频率段的上下限
        #         # 对当前窗口数据进行滤波
        #         data_filt = mne.filter.filter_data(data, self.fs, l_freq=low_freq, h_freq=high_freq, verbose=False)
        #         filtered_data[freq_idx] = data_filt
        #     # Drop start and end
        #     assert filtered_data.shape[-1] > 2 * self.fs, (f'eeg too short! {filtered_data.shape}')
        #     filtered_data = filtered_data[..., int(self.fs):-int(self.fs)]
        #     return filtered_data

        # def extract_de_features(self, data_window):
        #     # data_window[n_freqs, *], data of each band
        #     # 提取单个窗口的DE特征
        #     n_freqs = len(self.freqs)  # 获取频率段数量
        #
        #     de_features = np.zeros((data_window.shape[1], n_freqs))  # 初始化DE特征数组
        #
        #     # 遍历每个频率段
        #     for freq_idx, freq_range in enumerate(self.freqs):
        #         # 计算当前频率段的DE特征
        #         data_window_filt = data_window[freq_idx]
        #         de_one = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(data_window_filt, axis=1, keepdims=True)))
        #         de_features[:, freq_idx] = de_one.squeeze()  # 将DE特征存储在对应的频率段列中
        #
        #     return de_features  # 返回DE特征数组
        # process_and_save_eeg
        def sliding_window_extract_de(self,save_folder, total_id,train_ids, valid_ids):

            # 确定当前ID属于哪个数据集(预训练时需要)
            if str(total_id) in train_ids:
                dataset_type = "train"
            elif str(total_id) in valid_ids:
                dataset_type = "valid"
            else:
                print(f"警告: ID {total_id} 既不在训练集也不在验证集中")

                unclassified_txt_path = "unclassified/unclassified_txt"
                # 确保目标文件夹存在
                os.makedirs(os.path.dirname(unclassified_txt_path), exist_ok=True)
                # 将未分类ID追加到txt文件中
                with open(unclassified_txt_path, 'a') as f:
                    f.write(f"{str(total_id)}\n")

                return 0

            # dataset_type = 'train'
            # 获取原始数据
            eeg_data_array = np.array([self.eeg_data[chn] for chn in self.eeg_channels if
                                       chn in self.eeg_data and self.eeg_data[chn] is not None and chn != 'Trigger'])

            # 获取通道名称列表,排除'Trigger'通道
            raw_channels = [chn for chn in self.eeg_channels if
                        chn in self.eeg_data and self.eeg_data[chn] is not None and chn != 'Trigger'
                        ]

            #重采样到256hz
            orig_fs = self.fs  # 原始采样率
            target_fs = 256  # 目标采样率

            # 计算数据时长(秒)
            data_duration = eeg_data_array.shape[1] / self.fs

            # 如果数据长度小于4秒，则跳过处理
            if data_duration < 4.1:
                print(f"数据时长({data_duration:.2f}秒)小于4秒，跳过处理")
                return 0

            if orig_fs != target_fs:
                n_samples = int(eeg_data_array.shape[1] * (target_fs / orig_fs))
                resampled_data = signal.resample(eeg_data_array, n_samples, axis=1)
            else:
                resampled_data = eeg_data_array

            # 归一化
            resampled_data = (resampled_data - np.mean(resampled_data, axis=1, keepdims=True)) / \
                             (np.std(resampled_data, axis=1, keepdims=True) + 1e-8)
            resampled_data = np.clip(resampled_data, -10, 10)

            # 按4s长度切分
            # 步长设置
            step_size = self.step #设置步长为3s
            window_size = self.window_size  # 4秒对应的采样点数
            window_samples = int(window_size * target_fs)  # 窗口大小对应的采样点数
            step_samples = int(step_size * target_fs)  # 步长对应的采样点数

            # print('window_size:',window_size)
            # 计算可以切分的窗口数量
            n_windows = (resampled_data.shape[1] - window_samples) // step_samples + 1


            # 通道对齐和保存
            standard_channels = ['FP1', 'FPZ', 'FP2',
                               'AF3', 'AF4',
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8',
                               'O1', 'OZ', 'O2', ]

            # # 添加通道统计变量
            existing_channels = []
            missing_channels = []
            extra_channels = []  # 多余的通道

            # 将原始通道名转换为大写以进行比较
            raw_channels_upper = [ch.upper() for ch in raw_channels]

            # 统计匹配和缺失的通道
            for standard_ch in standard_channels:
                if standard_ch in raw_channels_upper:
                    existing_channels.append(standard_ch)
                else:
                    missing_channels.append(standard_ch)

            # # 统计多余的通道
            # for raw_ch in raw_channels:
            #     if raw_ch.upper() not in standard_channels:
            #         extra_channels.append(raw_ch)
            #
            # print(f"\n=== 通道统计 ===")
            # print(f"原始数据通道数: {len(raw_channels)}")
            # print(f"标准通道数: {len(standard_channels)}")
            # print(f"匹配到的通道数: {len(existing_channels)}")
            # print(f"缺失的通道数: {len(missing_channels)}")
            # print(f"多余的通道数: {len(extra_channels)}")
            #
            # print("\n--- 详细通道信息 ---")
            # print("现存的通道 ({}):\n{}".format(len(existing_channels),
            #                                     ', '.join(existing_channels)))
            # print("\n缺失的通道 ({}):\n{}".format(len(missing_channels),
            #                                       ', '.join(missing_channels)))
            # print("\n多余的通道 ({}):\n{}".format(len(extra_channels),
            #                                       ', '.join(extra_channels)))


            for win_idx in range(n_windows):
                # 获取当前窗口数据
                start_idx = win_idx * step_samples
                end_idx = start_idx + window_samples
                window_data = resampled_data[:, start_idx:end_idx]

                # 创建58通道的数据矩阵,初始化为0
                aligned_data = np.zeros((len(standard_channels), window_samples))

                # 将现有通道的数据填充到对应位置
                for i, ch in enumerate(raw_channels):
                    if ch.upper() in standard_channels:
                        idx = standard_channels.index(ch.upper())
                        aligned_data[idx] = window_data[i]

                # 转换为tensor
                tensor_data = torch.from_numpy(aligned_data).float()

                # 根据ID确定保存路径
                if dataset_type == "train":
                    save_dir = os.path.join(save_folder, "TrainFolder/0/")
                else:
                    save_dir = os.path.join(save_folder, "ValidFolder/0/")

                os.makedirs(save_dir, exist_ok=True)

                # 保存数据
                save_path = os.path.join(save_dir, f"eeg_{total_id}win_{win_idx}.edf")
                torch.save(tensor_data, save_path)

            print(f"Processed {n_windows} windows of data")
            return n_windows

        # def sliding_window_extract_de(self, save_folder, total_id):
        #
        #     use_channels_names = self.use_channels_names
        #
        #     # 滑动窗口提取DE特征并保存为pkl文件
        #     # 获取每个通道的值,通道名称在real内并且其值不为None,并且通道名称也不为'Trigger'
        #
        #     eeg_data_array = np.array([self.eeg_data[chn] for chn in self.eeg_channels if
        #                                chn in self.eeg_data and self.eeg_data[chn] is not None and chn != 'Trigger'])
        #
        #     # 归一化整个数据序列
        #     eeg_data_array = (eeg_data_array - np.mean(eeg_data_array, axis=1, keepdims=True)) / np.std(eeg_data_array,
        #                                                                                                 axis=1,
        #                                                                                                 keepdims=True)
        #     # 对归一化后的数据进行裁剪,将数值限制在-10到10的范围内
        #     # if (np.abs(eeg_data_array) > 10).astype(np.float32).mean() > 1e-3:
        #     #     raise ValueError(f'too many outliers! {eeg_data_array.shape}, '
        #     #                      f'outlier ratio: {(np.abs(eeg_data_array) > 10).astype(np.float32).mean().item()}')
        #     eeg_data_array = np.clip(eeg_data_array, -10, 10)
        #
        #     # filter the whole data sequence first
        #     t0 = time.time()
        #     eeg_data_array = self.filter(eeg_data_array)  # [n_freqs, nchns, n_samples]
        #     t1 = time.time()
        #     # 获取通道数和每个通道的长度
        #     _, n_chns, n_samples = eeg_data_array.shape
        #
        #     # 脑电标签，脑电设定为1
        #     labels = np.ones(n_chns, dtype=int)
        #     # print('label:', labels.shape)
        #     # print(labels)
        #     # 获取通道名称列表,排除'Trigger'通道
        #     channels = [chn for chn in self.eeg_channels if
        #                 chn in self.eeg_data and self.eeg_data[chn] is not None and chn != 'Trigger'
        #                 ]
        #
        #     # 获取每个通道的位置信息，channels_local数组的维度是[通道数，3（三个坐标点）]
        #     channels_local = np.array([self.eeg_chn_locs[chn] for chn in channels])
        #
        #     # print(channels_local[0])
        #     # 计算单个窗口的采样点数
        #     window_size_samples = int(self.window_size * self.fs)
        #     # 计算单个de特征提取窗口的采样点数
        #     de_window_size_samples = int(self.de_window_size * self.fs)
        #
        #     # 计算总窗口数量
        #     n_windows = (n_samples - window_size_samples) // (self.step * int(self.fs)) + 1
        #
        #     # 初始化DE特征列表
        #     all_de_features = []
        #
        #     # 滑动窗口提取DE特征并保存
        #     # 循环处理每个窗口
        #     for window in range(n_windows):
        #
        #         start = int(window * self.step * self.fs)  # 计算当前窗口的起始位置
        #         end = int(start + window_size_samples)  # 计算当前窗口的结束位置
        #
        #         # 提取所有通道的单个窗口的数据
        #         data_window = eeg_data_array[..., start:end]
        #
        #
        #         de_window = np.zeros((n_chns, window_size_samples // de_window_size_samples, len(self.freqs)))
        #         # 提取DE特征(使用向量化操作同时处理所有通道)
        #         for start_de in range(0, window_size_samples, de_window_size_samples):
        #             end_de = start_de + de_window_size_samples
        #             data_de_window = data_window[..., start_de:end_de]
        #             de_window[:, start_de // de_window_size_samples] = self.extract_de_features(data_de_window)
        #         # =====================================================================
        #
        #         # 保存为pkl文件
        #         filename = f'{total_id}_eeg_de_features_window_{window}.pkl'  # 构建文件名
        #         channel_filename = f'{total_id}_eeg_channel.pkl'
        #         label_filename = f'{total_id}_eeg_label.pkl'
        #
        #         # 保存路径
        #         save_path = os.path.join(save_folder, str(total_id))
        #
        #         # 创建保存路径,每个被试单独创建一个文件夹
        #         if not os.path.exists(save_path):
        #             os.makedirs(save_path)
        #
        #         filepath = os.path.join(save_path, filename)  # 构建文件路径
        #         with open(filepath, 'wb') as f:
        #             pickle.dump(de_window, f)  # 将DE特征保存为pkl文件
        #
        #         channel_filepath = os.path.join(save_path, channel_filename)
        #         with open(channel_filepath, 'wb') as f:
        #             pickle.dump(channels_local, f)  # 将通道位置保存为pkl文件
        #
        #         label_filepath = os.path.join(save_path, label_filename)
        #         with open(label_filepath, 'wb') as f:
        #             pickle.dump(labels, f)  # 将通道位置保存为pkl文件
        #
        #         # all_de_features.append(de_window)  # 将当前窗口的DE特征添加到列表中(这是所有的de特征，不是单个窗口的)
        #         # return de_window, channels
        #
        #     t2 = time.time()
        #     print(f'eeg process takes {t1-t0}s in filter and {t2-t1}s in DE extraction, save {n_windows} windows')
        #     return all_de_features, channels_local  # 返回所有窗口的DE特征和通道名称列表

    class FNIRS_data:
        def __init__(self, device, channels, data, note):
            self.fnirs_device = device
            self.fnirs_channels = channels  # 这个数据真实有效的近红外通道 {'(20.2,102.2,120)','()'}
            self.fnirs_chn_locs = {}  # 所有数据综合的近红外通道信息
            self.fnirs_data = data  # 真实的数据信息
            self.dataset_fnirs_note = note  # trigger含义，可以后期补

            # 后面加的
            self.fs = 17  # 采样率(2Hz)
            self.freqs = [0.01, 0.2]  # 指定的频率提取范围
            self.window_size = 10  # 单个切片的窗口大小(10秒)
            self.de_window_size = 1  # 提取DE特征的窗口大小(1秒)
            self.step = 5  # 切片窗口移动的步长(5秒)


        def sliding_window_extract_fe(self, save_folder, total_id,HBT=False):

            #单独处理每个通道
            if HBT == False:
                # 滑动窗口提取DE特征并保存为pkl文件
                # 获取每个通道的值,通道名称在real内并且其值不为None,并且通道名称也不为'Trigger'
                fnirs_data_array = np.array([self.fnirs_data[chn] for chn in self.fnirs_channels if
                                             chn in self.fnirs_data and self.fnirs_data[
                                                 chn] is not None and chn != 'Trigger'])
                # 获取通道名称列表,排除'Trigger'通道
                channels = [chn for chn in self.fnirs_channels if
                            chn in self.fnirs_data and self.fnirs_data[chn] is not None and chn != 'Trigger']
                # print(channels[0])
                #通道位置保存
                channels_local = []
                for chn in channels:
                    # 使用ast.literal_eval将字符串转换为tuple
                    coords = ast.literal_eval(chn)
                    # fnirs位置数据的第四个没有用，只保留前三个
                    channels_local.append(coords[:3])

                # 将列表转换为numpy数组
                channels_local = np.array(channels_local)
            #计算HBT=HBO-HBR，合并相同位置的通道
            else:
                merged_channels = {}
                incomplete_channels = []
                incomplete_folder = 'incomplete_FNIRS'  # HBO或HBR数据不完整的
                channels_local=[]
                for chn in self.fnirs_channels:
                    if chn in self.fnirs_data and self.fnirs_data[chn] is not None and chn != 'Trigger':
                        coords = ast.literal_eval(chn)
                        loc = tuple(coords[:3])
                        label = coords[3]
                        if loc not in merged_channels:
                            merged_channels[loc] = {'HBO': None, 'HBR': None}
                        merged_channels[loc][label] = self.fnirs_data[chn]

                # 获取合并后的通道数据
                fnirs_data_array = []
                for loc, data in merged_channels.items():
                    # 通道位置保存
                    channels_local.append(loc)
                    if data['HBO'] is not None and data['HBR'] is not None:
                        merged_data = data['HBO'] - data['HBR']
                        fnirs_data_array.append(merged_data)
                    else:
                        incomplete_channels.append(loc)
                fnirs_data_array = np.array(fnirs_data_array)
                channels_local = np.array(channels_local)
                # 检查是否有不完整的通道数据,即只有HBO或者HBR
                if len(incomplete_channels) > 0:
                    # 将不完整通道的ID保存到指定文件夹
                    incomplete_path = os.path.join(incomplete_folder, str(total_id))
                    if not os.path.exists(incomplete_path):
                        os.makedirs(incomplete_path)
                    incomplete_filename = f'{total_id}_incomplete_channels.pkl'
                    incomplete_filepath = os.path.join(incomplete_path, incomplete_filename)
                    with open(incomplete_filepath, 'wb') as f:
                        pickle.dump(incomplete_channels, f)


            # 归一化整个数据序列
            fnirs_data_array = (fnirs_data_array - np.mean(fnirs_data_array, axis=1, keepdims=True)) / np.std(fnirs_data_array,
                                                                                                        axis=1,
                                                                                                       keepdims=True)
            # 对归一化后的数据进行裁剪,将数值限制在-10到10的范围内
            fnirs_data_array = np.clip(fnirs_data_array, -10, 10)

            # 对整个数据序列进行滤波
            # fnirs_data_array = mne.filter.filter_data(fnirs_data_array, self.fs, l_freq=self.freqs[0],
            #                                           h_freq=self.freqs[1])
            assert fnirs_data_array.shape[-1] > 2 * self.fs, (f'fnirs too short! {fnirs_data_array.shape}')
            fnirs_data_array = fnirs_data_array[..., 2 * int(self.fs):]  # fixme: drop start to offset fnirs_data_array

            # 获取通道数和每个通道的长度
            n_chns, n_samples = fnirs_data_array.shape


            # 脑电标签,fnirs设定为0
            labels = np.zeros(n_chns, dtype=int)
            # print('label:', labels.shape)
            # print(labels)


            # 计算单个窗口的采样点数
            window_size_samples = int(self.window_size * self.fs)

            # 计算总窗口数量
            n_windows = int((n_samples - window_size_samples) // (self.step * int(self.fs))) + 1

            # 初始化特征列表
            all_de_features = []

            # 滑动窗口提取DE特征并保存
            # 循环处理每个窗口
            for window in range(n_windows):
                start = int(window * self.step * self.fs)  # 计算当前窗口的起始位置
                end = start + window_size_samples  # 计算当前窗口的结束位置
                # print(f"start: {start}, type: {type(start)}")
                # print(f"end: {end}, type: {type(end)}")
                # start = int(start)
                # end = int(end)

                # 提取所有通道的单个窗口的数据
                data_window = fnirs_data_array[:, start:end]


                # 对当前窗口进行降采样
                downsampled_window = data_window

                # 获取降采样后的长度
                downsampled_len = downsampled_window.shape[1]  # 降采样后的长度

                # 对每个通道进行插值
                n_chns = downsampled_window.shape[0]

                # =========================================================================
                # new_len = 50
                # interpolated_data = signal.resample(downsampled_window, new_len, axis=-1)
                # -----------------------------------------------------------------------
                interpolated_data = np.zeros((n_chns, 100))
                # print('interpolate', downsampled_window.shape, interpolated_data.shape)
                for i in range(n_chns):
                    # 对第i个通道的数据进行插值
                    old_len = downsampled_len
                    new_len = 100
                    interpolated_data[i, :] = signal.resample(downsampled_window[i, :], new_len)
                # ==============================================================================
                # print(interpolated_data.shape)
                # 重构downsampled_window为 (n_chns, window_size_samples // de_window_size_samples, 5)
                n_chns, n_samples = interpolated_data.shape
                n_segments = n_samples // 10
                reshaped_window = interpolated_data.reshape(n_chns, n_segments, 10)

                # 保存为pkl文件
                filename = f'{total_id}_fnirs_de_features_window_{window}.pkl'  # 构建文件名
                channel_filename = f'{total_id}_fnirs_channel.pkl'
                label_filename = f'{total_id}_fnirs_label.pkl'

                # 保存路径
                save_path = os.path.join(save_folder, str(total_id))

                # 创建保存路径,每个被试单独创建一个文件夹
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                filepath = os.path.join(save_path, filename)  # 构建文件路径
                with open(filepath, 'wb') as f:
                    pickle.dump(reshaped_window, f)  # 将降采样后的数据保存为pkl文件

                channel_filepath = os.path.join(save_path, channel_filename)
                with open(channel_filepath, 'wb') as f:
                    pickle.dump(channels_local, f)  # 将通道位置保存为pkl文件

                label_filepath = os.path.join(save_path, label_filename)
                with open(label_filepath, 'wb') as f:
                    pickle.dump(labels, f)  # 将通道位置保存为pkl文件

                # all_de_features.append(downsampled_window)  # 将当前窗口的降采样数据添加到列表中

            return all_de_features, channels_local  # 返回所有窗口的降采样数据和通道名称列表
            # # 滑动窗口提取DE特征并保存为pkl文件
            # # 获取每个通道的值,通道名称在real内并且其值不为None,并且通道名称也不为'Trigger'
            # eeg_data_array = np.array([self.fnirs_data[chn] for chn in self.fnirs_channels if
            #                            chn in self.fnirs_data and self.fnirs_data[
            #                                chn] is not None and chn != 'Trigger'])
            # # 获取通道数和每个通道的长度
            # n_chns, n_samples = eeg_data_array.shape
            #
            # # 脑电标签，fnirs设定为0
            # labels = np.zeros(n_chns, dtype=int)
            # print('label:', labels.shape)
            # print(labels)
            #
            # # 获取通道名称列表,排除'Trigger'通道
            # channels = [chn for chn in self.fnirs_channels if
            #             chn in self.fnirs_data and self.fnirs_data[chn] is not None and chn != 'Trigger']
            # print(channels[0])
            #
            # channels_local = []
            # for chn in channels:
            #     # 使用ast.literal_eval将字符串转换为tuple
            #     coords = ast.literal_eval(chn)
            #     channels_local.append(coords)
            #
            # # 将列表转换为numpy数组
            # channels_local = np.array(channels_local)
            #
            # print(channels_local[1])
            #
            # # 计算单个窗口的采样点数
            # window_size_samples = self.window_size * self.fs
            #
            # # 计算单个de特征提取窗口的采样点数
            # de_window_size_samples = self.de_window_size * self.fs
            #
            # # 计算总窗口数量
            # n_windows = (n_samples - window_size_samples) // (self.step * self.fs) + 1
            #
            # # 初始化DE特征列表
            # all_de_features = []
            #
            # # 滑动窗口提取DE特征并保存
            # # 循环处理每个窗口
            # for window in range(n_windows):
            #
            #     start = window * self.step * self.fs  # 计算当前窗口的起始位置
            #     end = start + window_size_samples  # 计算当前窗口的结束位置
            #
            #     # 提取所有通道的单个窗口的数据
            #     data_window = eeg_data_array[:, start:end]
            #
            #     # 提取DE特征
            #     # 初始化单个窗口的de特征
            #     de_window = np.zeros((n_chns, window_size_samples // de_window_size_samples, len(self.freqs)))
            #
            #     # 提取DE特征(使用向量化操作同时处理所有通道)
            #     for start_de in range(0, window_size_samples, de_window_size_samples):
            #         end_de = start_de + de_window_size_samples
            #         data_de_window = data_window[:, start_de:end_de]
            #         de_window[:, start_de // de_window_size_samples] = self.extract_de_features(data_de_window)
            #
            #     # 保存为pkl文件
            #     filename = f'{total_id}_fnirs_de_features_window_{window}.pkl'  # 构建文件名
            #     channel_filename = f'{total_id}_fnirs_channel.pkl'
            #     label_filename = f'{total_id}_fnirs_label.pkl'
            #     # 保存路径
            #     save_path = os.path.join(save_folder, total_id)
            #
            #     # 创建保存路径,每个被试单独创建一个文件夹
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #
            #     filepath = os.path.join(save_path, filename)  # 构建文件路径
            #     with open(filepath, 'wb') as f:
            #         pickle.dump(de_window, f)  # 将DE特征保存为pkl文件
            #
            #     channel_filepath = os.path.join(save_path, channel_filename)
            #     with open(channel_filepath, 'wb') as f:
            #         pickle.dump(channels_local, f)  # 将通道位置保存为pkl文件
            #
            #     label_filepath = os.path.join(save_path, label_filename)
            #     with open(label_filepath, 'wb') as f:
            #         pickle.dump(labels, f)  # 将通道位置保存为pkl文件
            #
            #     # all_de_features.append(de_window)  # 将当前窗口的DE特征添加到列表中(这是所有的de特征，不是单个窗口的)
            #     # return de_window,channels
            # return all_de_features, channels_local  # 返回所有窗口的DE特征和通道名称列表

    class Preprocess_info:
        def __init__(self):
            # EEG预处理留档
            self.eeg_preprocess_pipeline = []
            self.eeg_freq = None
            self.eeg_delete_chns = []
            self.eeg_bad_channels = []
            self.eeg_freq_bands1 = []
            self.eeg_ica_or_no = 'no'
            self.eeg_rereference = 'common average'
            self.eeg_freq_bands2 = []
            self.eeg_note = None

            # fNIRS预处理留档
            self.fnirs_preprocess_pipeline = []
            self.fnirs_freq = None
            self.fnirs_delete_chns = []  # 实际上是插值的通道
            self.fnirs_freq_bands1 = []
            self.fnirs_note = None

    def conbine_eeg_fnirs(self, save_folder, total_id):

        save_path = os.path.join(save_folder, str(total_id))

        # 获取eeg和fnirs数据的文件路径
        eeg_files = [f for f in os.listdir(save_path) if f.startswith(f'{total_id}_eeg_de_features')]
        fnirs_files = [f for f in os.listdir(save_path) if f.startswith(f'{total_id}_fnirs_de_features')]

        # 确保eeg和fnirs文件数量相同
        # assert len(eeg_files) == len(fnirs_files), "EEG and fNIRS file counts do not match"

        # 确定数量少的是eeg还是fnirs
        if len(eeg_files) < len(fnirs_files):
            min_files = eeg_files
            max_files = fnirs_files
            print(
                f"EEG文件数量少于fNIRS文件数量,分别为{len(eeg_files)}和{len(fnirs_files)},少了{len(fnirs_files) - len(eeg_files)}个文件")
        elif len(eeg_files) > len(fnirs_files):
            min_files = fnirs_files
            max_files = eeg_files
            print(
                f"fNIRS文件数量少于EEG文件数量,分别为{len(fnirs_files)}和{len(eeg_files)},少了{len(eeg_files) - len(fnirs_files)}个文件")
        else:
            min_files = eeg_files

        # 合并eeg和fnirs数据
        for min_file in min_files:

            # 从文件名中提取窗口索引
            window_index = int(min_file.split('_')[-1].split('.')[0])

            # 根据窗口索引找到对应的eeg和fnirs文件
            eeg_file = f'{total_id}_eeg_de_features_window_{window_index}.pkl'
            fnirs_file = f'{total_id}_fnirs_de_features_window_{window_index}.pkl'
            eeg_path = os.path.join(save_path, eeg_file)
            fnirs_path = os.path.join(save_path, fnirs_file)

            with open(eeg_path, 'rb') as f:
                eeg_data = pickle.load(f)

            with open(fnirs_path, 'rb') as f:
                fnirs_data = pickle.load(f)

            combined_data = np.concatenate((eeg_data, fnirs_data), axis=0)

            combined_filename = f'{total_id}_eeg_fnirs_de_features_window_{window_index}.pkl'
            combined_path = os.path.join(save_path, combined_filename)
            with open(combined_path, 'wb') as f:
                pickle.dump(combined_data, f)

        # 删除原始的EEG和fNIRS文件
        for file in eeg_files + fnirs_files:
            file_path = os.path.join(save_path, file)
            os.remove(file_path)
        # # 合并eeg和fnirs数据
        # for i, (eeg_file, fnirs_file) in enumerate(zip(eeg_files, fnirs_files)):
        #     eeg_path = os.path.join(save_path, eeg_file)
        #     fnirs_path = os.path.join(save_path, fnirs_file)
        #
        #     with open(eeg_path, 'rb') as f:
        #         eeg_data = pickle.load(f)
        #
        #     with open(fnirs_path, 'rb') as f:
        #         fnirs_data = pickle.load(f)
        #
        #     combined_data = np.concatenate((eeg_data, fnirs_data), axis=0)
        #
        #     combined_filename = f'{total_id}_eeg_fnirs_de_features_window_{i}.pkl'
        #     combined_path = os.path.join(save_path, combined_filename)
        #     with open(combined_path, 'wb') as f:
        #         pickle.dump(combined_data, f)
        #
        #     # 删除原始eeg和fnirs文件
        #     os.remove(eeg_path)
        #     os.remove(fnirs_path)

        # 合并通道文件
        eeg_channel_path = os.path.join(save_path, f'{total_id}_eeg_channel.pkl')
        fnirs_channel_path = os.path.join(save_path, f'{total_id}_fnirs_channel.pkl')

        with open(eeg_channel_path, 'rb') as f:
            eeg_channels = pickle.load(f)

        with open(fnirs_channel_path, 'rb') as f:
            fnirs_channels = pickle.load(f)

        combined_channels = np.concatenate((eeg_channels, fnirs_channels), axis=0)

        combined_channel_path = os.path.join(save_path, f'{total_id}_eeg_fnirs_channel.pkl')
        with open(combined_channel_path, 'wb') as f:
            pickle.dump(combined_channels, f)

        # 删除原始通道文件
        os.remove(eeg_channel_path)
        os.remove(fnirs_channel_path)

        # 合并标签文件
        eeg_label_path = os.path.join(save_path, f'{total_id}_eeg_label.pkl')
        fnirs_label_path = os.path.join(save_path, f'{total_id}_fnirs_label.pkl')

        with open(eeg_label_path, 'rb') as f:
            eeg_labels = pickle.load(f)

        with open(fnirs_label_path, 'rb') as f:
            fnirs_labels = pickle.load(f)

        combined_labels = np.concatenate((eeg_labels, fnirs_labels), axis=0)

        combined_label_path = os.path.join(save_path, f'{total_id}_eeg_fnirs_label.pkl')
        with open(combined_label_path, 'wb') as f:
            pickle.dump(combined_labels, f)

        # 删除原始标签文件
        os.remove(eeg_label_path)
        os.remove(fnirs_label_path)


def set_affinity(process_id, num_cores):
    p = psutil.Process(os.getpid())  # Get the current process
    # Assign the CPU core, wrap around if process_id exceeds available cores
    core_id = process_id % num_cores
    p.cpu_affinity([core_id])  # Set the process to only run on the assigned core


def process_subject(file_path, save_folder,train_ids,valid_ids, check_pickle=False):
    total_id = int(file_path.split('/')[-1].replace('sub_', ''))
    set_affinity(total_id, 128)

    try:
        with open(file_path, 'rb') as f:
            subject = pickle.load(f)
        if os.path.exists(os.path.join(save_folder, str(total_id))):  # fixme
            if check_pickle:
                pickle_error_flag = False
                for file in os.listdir(os.path.join(save_folder, str(total_id))):
                    try:
                        with open(os.path.join(save_folder, str(total_id), file), 'rb') as f:
                            _ = pickle.load(f)
                    except pickle.PickleError:
                        pickle_error_flag = True
                        break
                if not pickle_error_flag:
                    return
            else:
                return

        if int(subject.subject_total_id) != total_id:
            print('wrong total id', subject.subject_total_id, total_id)
        pid = os.getpid()
        cpu_list = os.sched_getaffinity(pid)
        print('process', pid, 'total_id', total_id, 'run in cpu:', cpu_list)
        from types import MethodType
        subject.conbine_eeg_fnirs = MethodType(Subject_data.conbine_eeg_fnirs,subject)
        if subject.eeg_info == 'yes':
            # print(list(vars(subject.eeg).keys()))

            subject.eeg.sliding_window_extract_de = MethodType(Subject_data.EEG_data.sliding_window_extract_de,
                                                               subject.eeg)
            # subject.eeg.extract_de_features = MethodType(Subject_data.EEG_data.extract_de_features, subject.eeg)
            # subject.eeg.filter = MethodType(Subject_data.EEG_data.filter, subject.eeg)
            subject.eeg.window_size = 10  #修改window_size的时候在这里改
            subject.eeg.fs = subject.preprocess.eeg_freq
            subject.eeg.de_window_size = 1  # fixme: tunable
            subject.eeg.eeg_chn_locs = eeg_locs  # todo: copy from Preprocess
            if '/SLEEP/' in file_path:
                subject.eeg.step = 30  # todo
            else:
                subject.eeg.step = 5
            # subject.eeg.freqs = [[1, 4], [4, 8], [8, 14], [14, 30], [30, 47]]
            subject.eeg.freqs = [[1,3],[3,5],[5,8],[8,10],[10,12],[12,16],[16,20],[20,30],[30,40],[40,47]]

        if subject.fnirs_info == 'yes':
            # print(list(vars(subject.fnirs).keys()))
            # from types import MethodType
            subject.fnirs.sliding_window_extract_fe = MethodType(Subject_data.FNIRS_data.sliding_window_extract_fe,
                                                               subject.fnirs)
            subject.fnirs.window_size = 10
            subject.fnirs.fs = subject.preprocess.fnirs_freq
            subject.fnirs.de_window_size = 1  # todo: tunable
            subject.fnirs.step = 5
            subject.fnirs.freqs = [0.01, 0.2]

        print('to process', 'label', subject.subject_label,
              'dataset', subject.subject_dataset,
              'total id', subject.subject_total_id,
              'subject id', subject.subject_id_dataset)
        if subject.eeg is not None:
            subject.eeg.sliding_window_extract_de(save_folder, total_id,train_ids, valid_ids)

        #     raise NotImplementedError
        
        
        # # print(subject.fnirs.fnirs_channels)
        # if subject.fnirs is not None:
        #     #HBT=True时合并相同位置通道，计算HBT=HBO-HBR作为通道的值，HBT=False时，则单独处理每个通道。
        #     subject.fnirs.sliding_window_extract_fe(save_folder, total_id, HBT=False)
        #     # raise NotImplementedError
        # if (subject.eeg is not None) and (subject.fnirs is not None):
        #     subject.conbine_eeg_fnirs(save_folder, total_id)
        #     # raise NotImplementedError

        # Save subject info
        info_filename = f'{total_id}_info.pkl'
        ses_mark=subject.subject_file_name.split('/')[-1]
        info = {'subject_label': subject.subject_label, 'dataset': subject.subject_dataset, 'subject_id_dateset' : subject.subject_id_dataset,'subject_file_name':subject.subject_file_name,'ses_mark':ses_mark}
        if os.path.exists(os.path.join(save_folder, str(total_id))):
            with open(os.path.join(save_folder, str(total_id), info_filename), 'wb') as f:
                pickle.dump(info, f)

    except (AssertionError, ValueError) as e:
        logging.error("An error occurred during multiprocessing: %s. Subject ID: %s, Total ID: %s", e, subject.subject_total_id, total_id)
        path_to_remove = os.path.join(save_folder, str(total_id))
        if os.path.exists(path_to_remove):
            os.system(f'rm -r {path_to_remove}')
            logging.info(f"Successfully removed {path_to_remove}")
        else:
            logging.warning(f"Path does not exist: {path_to_remove}")
    except Exception as e:
        logging.error("An error occurred during multiprocessing: %s. Subject ID: %s, Total ID: %s", e,
                      subject.subject_total_id, total_id)
        path_to_remove = os.path.join(save_folder, str(total_id))
        if os.path.exists(path_to_remove):
            os.system(f'rm -r {path_to_remove}')
            logging.info(f"Successfully removed {path_to_remove}")
        else:
            logging.warning(f"Path does not exist: {path_to_remove}")


def get_dataset_ids(train_folder_path, valid_folder_path):
    """
    从两个父文件夹中读取子文件夹名作为训练集和验证集的ID列表

    Parameters:
        train_folder_path: 训练集父文件夹路径
        valid_folder_path: 验证集父文件夹路径

    Returns:
        train_ids: 训练集ID列表
        valid_ids: 验证集ID列表
    """
    # 获取训练集文件夹中的子文件夹名
    train_ids = [str(folder) for folder in os.listdir(train_folder_path)
                 if os.path.isdir(os.path.join(train_folder_path, folder))]

    # 获取验证集文件夹中的子文件夹名
    valid_ids = [str(folder) for folder in os.listdir(valid_folder_path)
                 if os.path.isdir(os.path.join(valid_folder_path, folder))]

    # 检查是否有重复的ID
    duplicate_ids = set(train_ids) & set(valid_ids)
    if duplicate_ids:
        print(f"警告: 在训练集和验证集中发现重复的ID: {duplicate_ids}")

    # 打印统计信息
    print("\n=== 数据集划分统计 ===")
    print(f"训练集样本数: {len(train_ids)}")
    print(f"验证集样本数: {len(valid_ids)}")
    print(f"总样本数: {len(train_ids) + len(valid_ids)}")

    return train_ids, valid_ids


if __name__ == '__main__':
    setup_logger()
    # Pretrain Data ===================================================

    # Data processing for the directories to be preprocessed
    data_folders = [
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/SEED',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/LEMON',
        # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/HBN_1',
        # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/HUAWEI_EEG',
        '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/SLEEP/HMSP_PROCESSED/HMSP_data',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/train/abnormal',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/train/normal',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_events/eval',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_events/train',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_epilepsy/00_epilepsy',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_epilepsy/01_no_epilepsy',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_slowing',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/SXMU_1_PROCESSED/HC',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/SXMU_1_PROCESSED/MDD',
        # '/home/wangkuiyu/data1/LEM/new_data_pool/fNIRS/COMPETE_PROCESSED',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG_fNIRS/ZD_fusion_1',

        '/home/wangkuiyu/data1/LEM/new_data_pool/CSA-PD-W',
        '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/E-CAM-S',
        '/home/wangkuiyu/data1/LEM/new_data_pool/TBI_01',
        '/home/wangkuiyu/data1/LEM/new_data_pool/TBI_02',
        '/home/wangkuiyu/data1/LEM/new_data_pool/TBI_03',

        # 新加入到预训练的数据
        # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/AD_FD_HC_PROCESSED',  sft
        '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PREDICT/PREDICT-mTBI_Rest',
        '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PREDICT/PREDICT-OCD_Flanker',
        # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/Porn-addiction',  sft
        '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PD_Gait',
        # '/home/wangkuiyu/data1/LEM/new_data_pool/First_Episode_Psychosis_Control_1', sft
        # '/home/wangkuiyu/data1/LEM/new_data_pool/First_Episode_Psychosis_Control_2', sft
        # '/home/wangkuiyu/data1/LEM/new_data_pool/QDHSM',  sft
        '/home/wangkuiyu/data1/LEM/new_data_pool/One_person_microstate/',
        '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/52_Resting/',
        '/home/wangkuiyu/data1/LEM/new_data_pool/51_FACED/',

    ]

    # 设置父文件夹路径
    train_folder = "/data1/wangkuiyu/LEM/MultiModel/pretrain_set_with_origin_train/"
    valid_folder = "/data1/wangkuiyu/LEM/MultiModel/pretrain_set_with_origin_val/"
    # file_paths = []
    # 获取训练集和验证集ID
    train_ids, valid_ids = get_dataset_ids(train_folder, valid_folder)


    for data_folder in data_folders:
        file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith('sub_')]

        # Use a partial function to include save_folder during processing
        process_subject_partial = partial(process_subject, save_folder='./EEGPT_pretrain_set_origin_114', check_pickle=True,train_ids=train_ids, valid_ids=valid_ids)
        with Pool(processes=16) as pool:
            pool.map(process_subject_partial, file_paths)

    # file_path = "/data1/wangkuiyu/LEM/resting_data_pool/resting_eye_open/60_ANDing/sub_104716"
    # process_subject(file_path=file_path,save_folder='./pretrain_set_10bands',check_pickle=True,train_ids=train_ids, valid_ids=valid_ids)

    # # Downstream for test.py dataset processing (directories marked as "leave for downstream")
    # downstream_data_folders = [
    #     # '/data1/wangkuiyu/LEM/resting_data_pool/resting_eye_close/52_Resting_state',
    #     # '/data1/wangkuiyu/LEM/resting_data_pool/resting_eye_open/52_Resting_state',
    #     '/data1/wangkuiyu/LEM/resting_data_pool/resting_eye_close/56-SRM_resting',
    #     '/data1/wangkuiyu/LEM/resting_data_pool/resting_eye_close/62_leixu_test_retest',
    #     '/data1/wangkuiyu/LEM/resting_data_pool/resting_eye_open/62_leixu_test_retest',
    #     '/data1/wangkuiyu/LEM/resting_data_pool/resting_eye_close/63_Sleep_Deprivation',
    #     '/data1/wangkuiyu/LEM/resting_data_pool/resting_eye_open/63_Sleep_Deprivation',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/Td_eyeopen',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/Td_eyeclose',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/Parkinson_eyes_open_PROCESSED',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/AD_FD_HC_PROCESSED',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PREDICT/PREDICT-mTBI_Rest',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PREDICT/PREDICT-Depression_Rest',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PREDICT/PREDICT-OCD_Flanker',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PREDICT/PREDICT-Schizophrenia_Conflict',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PREDICT/PREDICT-Depression_RL',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PREDICT/PREDICT-PD_LPC_Rest',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PREDICT/PREDICT-PD_LPC_Rest_2',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/eval/abnormal',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/eval/normal',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/train/abnormal',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/train/normal',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/SXMU_2_PROCESSED/HC',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/SXMU_2_PROCESSED/MDD',
    #     #
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/CSA-PD-W',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/Parkinson',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/Porn-addiction',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/E-CAM-S',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/EEG_in_SZ',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SFT/PD_Gait',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/TBI_01',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/TBI_02',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/TBI_03',
    #     #
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/QDHSM',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/SXMU-ERP',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/First_Episode_Psychosis_Control_1',
    #     # '/home/wangkuiyu/data1/LEM/new_data_pool/First_Episode_Psychosis_Control_2',
    #     #
    #     # # Resting
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/LEMON',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_open/Td_eyeopen',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_close/Td_eyeclose',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_open/Parkinson_eyes_open_PROCESSED',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_close/AD_FD_HC_PROCESSED',
    #     # '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/PREDICT-Depression_Rest',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/PREDICT-PD_LPC_Rest',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_open/PREDICT-PD_LPC_Rest_2',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/SXMU_2_PROCESSED/HC',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/SXMU_2_PROCESSED/MDD',
    #     #
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/CSA-PD-W',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/Parkinson',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_close/Porn-addiction',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_open/Porn-addiction',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_close/EEG_in_SZ',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/First_Episode_Psychosis_Control_1',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/First_Episode_Psychosis_Control_2',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_open/QDHSM/',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_open/SXMU-ERP/',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_close/QDHSM/',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_close/SXMU-ERP/',
    #     # '/home/wangkuiyu/data1/LEM/clinical/YF_HUILONGGUAN/',
    #     #
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_close/HUAWEI_eye_close/',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_open/HUAWEI_eye_open/',
    #     # '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_open/ZD_fusion_eye_open/',
    #     # '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_close/ZD_fusion_eye_close/',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_unknown/57_Predict-Depression-Rest-New',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_close/60_ANDing/',
    #     '/home/wangkuiyu/data1/LEM/resting_data_pool/resting_eye_open/60_ANDing/',
    # ]

    # downstream_save_folders = [
    #     # './fine_pool_10bands/SFT/resting_eye_close/52_Resting_state',
    #     # './fine_pool_10bands/SFT/resting_eye_open/52_Resting_state',
    #     './fine_pool_10bands/SFT/resting_eye_close/56-SRM_resting',
    #     './fine_pool_10bands/SFT/resting_eye_close/62_leixu_test_retest',
    #     './fine_pool_10bands/SFT/resting_eye_open/62_leixu_test_retest',
    #     './fine_pool_10bands/SFT/resting_eye_close/63_Sleep_Deprivation',
    #     './fine_pool_10bands/SFT/resting_eye_open/63_Sleep_Deprivation',
    #     # './fine_pool_10bands/SFT/Td_eyeopen',
    #     # './fine_pool_10bands/SFT/Td_eyeclose',
    #     # './fine_pool_10bands/SFT/Parkinson_eyes_open_PROCESSED',
    #     # './fine_pool_10bands/SFT/AD_FD_HC_PROCESSED',
    #     # './fine_pool_10bands/SFT/PREDICT/PREDICT-mTBI_Rest',
    #     # './fine_pool_10bands/SFT/PREDICT/PREDICT-Depression_Rest',
    #     # './fine_pool_10bands/SFT/PREDICT/PREDICT-OCD_Flanker',
    #     # './fine_pool_10bands/SFT/PREDICT/PREDICT-Schizophrenia_Conflict',
    #     # './fine_pool_10bands/SFT/PREDICT/PREDICT-Depression_RL',
    #     # './fine_pool_10bands/SFT/PREDICT/PREDICT-PD_LPC_Rest',
    #     # './fine_pool_10bands/SFT/PREDICT/PREDICT-PD_LPC_Rest_2',
    #     # './fine_pool_10bands/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/eval/abnormal',
    #     # './fine_pool_10bands/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/eval/normal',
    #     # './fine_pool_10bands/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/train/abnormal',
    #     # './fine_pool_10bands/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/train/normal',
    #     # './fine_pool_10bands/EEG/SXMU_2_PROCESSED/HC',
    #     # './fine_pool_10bands/EEG/SXMU_2_PROCESSED/MDD',
    #     # #
    #     # './fine_pool_10bands/CSA-PD-W',
    #     # './fine_pool_10bands/SFT/Parkinson',
    #     # './fine_pool_10bands/SFT/Porn-addiction',
    #     # './fine_pool_10bands/SFT/E-CAM-S',
    #     # './fine_pool_10bands/SFT/EEG_in_SZ',
    #     # './fine_pool_10bands/SFT/PD_Gait',
    #     # './fine_pool_10bands/TBI_01',
    #     # './fine_pool_10bands/TBI_02',
    #     # './fine_pool_10bands/TBI_03',
    #     #
    #     # './fine_pool_10bands/QDHSM',
    #     # './fine_pool_10bands/SXMU-ERP',
    #     # './fine_pool_10bands/First_Episode_Psychosis_Control_1',
    #     # './fine_pool_10bands/First_Episode_Psychosis_Control_2',
    #     #
    #     # # Resting
    #     './resting_fine_pool_10bands/resting_unknown/LEMON',
    #     './resting_fine_pool_10bands/resting_eye_open/Td_eyeopen',
    #     './resting_fine_pool_10bands/resting_eye_close/Td_eyeclose',
    #     './resting_fine_pool_10bands/resting_eye_open/Parkinson_eyes_open_PROCESSED',
    #     './resting_fine_pool_10bands/resting_eye_close/AD_FD_HC_PROCESSED',
    #     # './resting_fine_pool_10bands/resting_unknown/PREDICT-Depression_Rest',
    #     './resting_fine_pool_10bands/resting_unknown/PREDICT-PD_LPC_Rest',
    #     './resting_fine_pool_10bands/resting_eye_open/PREDICT-PD_LPC_Rest_2',
    #     './resting_fine_pool_10bands/resting_unknown/SXMU_2_PROCESSED/HC',
    #     './resting_fine_pool_10bands/resting_unknown/SXMU_2_PROCESSED/MDD',
    #     #
    #     './resting_fine_pool_10bands/resting_unknown/CSA-PD-W',
    #     './resting_fine_pool_10bands/resting_unknown/Parkinson',
    #     './resting_fine_pool_10bands/resting_eye_close/Porn-addiction',
    #     './resting_fine_pool_10bands/resting_eye_open/Porn-addiction',
    #     './resting_fine_pool_10bands/resting_eye_close/EEG_in_SZ',
    #     './resting_fine_pool_10bands/resting_unknown/First_Episode_Psychosis_Control_1',
    #     './resting_fine_pool_10bands/resting_unknown/First_Episode_Psychosis_Control_2',
    #     './resting_fine_pool_10bands/resting_eye_open/QDHSM/',
    #     './resting_fine_pool_10bands/resting_eye_open/SXMU-ERP/',
    #     './resting_fine_pool_10bands/resting_eye_close/QDHSM/',
    #     './resting_fine_pool_10bands/resting_eye_close/SXMU-ERP/',
    #     # './resting_fine_pool_10bands/clinical/YF_HUILONGGUAN/',
    #     #
    #     './resting_fine_pool_10bands/resting_eye_close/HUAWEI_eye_close/',
    #     './resting_fine_pool_10bands/resting_eye_open/HUAWEI_eye_open/',
    #     # './resting_fine_pool_10bands/resting_eye_open/ZD_fusion_eye_open/',
    #     # './resting_fine_pool_10bands/resting_eye_close/ZD_fusion_eye_close/',
    #     './resting_fine_pool_10bands/resting_unknown/57_Predict-Depression-Rest-New',
    #     './resting_fine_pool_10bands/resting_eye_close/60_ANDing/',
    #     './resting_fine_pool_10bands/resting_eye_open/60_ANDing/',
    # ]

    # for data_folder, save_folder in zip(downstream_data_folders, downstream_save_folders):
    #     file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith('sub_')]
    #     process_subject_partial = partial(process_subject, save_folder=save_folder)
    #     with Pool(processes=128) as pool:
    #         pool.map(process_subject_partial, file_paths)


