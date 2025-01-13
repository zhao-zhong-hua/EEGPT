# Authors: Robert Wong <robert_is_here@163.com>
# Date : 2024.04.02
# from eeg_channels import *
# 一个预训练数据应该有的类型
class Subject_data():

    def __init__(self):
        self.subject_total_id = None # 被试在所有数据集里的编号
        self.subject_gender = None # 0 for female, 1 for male
        self.subject_age = None # the age of the subject
        self.subject_dataset = None # 数据集名称
        self.subject_id_dataset = None # 这个被试在这个数据集里的名称
        self.subject_dir_id = None # 被试的第几个文件夹？(任务)
        self.subject_file_name = None # 这个被处理的数据文件的原始名称
        self.subject_time_label = None
        self.subject_dataset_note = 'This is a dataset for LEM' # 用于概括数据集的作用
        self.subject_label = None  # 一些关于被试的标记
        self.file_format = None

        self.eeg_info = 'no' # 有EEG数据
        self.fnirs_info = 'no' # 有近红外数据
        self.fmri_info = 'no' # 用于未来数据集
        self.mri_info = 'no' # 用于未来数据集

        self.eeg = None # 具体的数据信息
        self.fnirs = None
        self.fmri = None
        self.mri = None
        self.preprocess = None
        self.add_preprocess_history() # 初始化参数

    def add_eeg_data(self,device, channels, data, note):
        self.eeg = self.EEG_data(device, channels,data,note)
        self.eeg_info = 'yes'

    def add_fnirs_data(self, device, channels,data,note):
        self.fnirs = self.FNIRS_data(device, channels,data,note)
        self.fnirs_info = 'yes'

    def add_preprocess_history(self):# 这一条全靠变量赋予
        self.preprocess = self.Preprocess_info()

    class EEG_data:
        def __init__(self,device, channels, data, note):
            self.eeg_device = device
            self.eeg_channels = channels # 这个被试真实有效的通道，不要参考通道
            self.eeg_chn_locs = eeg_locs # 128个通道的位置信息  {'Fz':(1,1),'Oz':(100,12)}
            self.eeg_data = data # 128个通道的信息  {'Fz':array, 'Oz':array, 'Cz':None, ..., 'Trigger': array}
            self.trigger_note = note # trigger含义 '0 for positive video, 1 for neutral video , 2 for negative'

    class FNIRS_data:
        def __init__(self,device, channels, data, note):
            self.fnirs_device = device
            self.fnirs_channels = channels # 这个数据真实有效的近红外通道 {'(20.2,102.2,120)','()'}
            self.fnirs_chn_locs = {} # 所有数据综合的近红外通道信息
            self.fnirs_data = data # 真实的数据信息
            self.dataset_fnirs_note = note # trigger含义，可以后期补

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
            self.fnirs_delete_chns = [] # 实际上是插值的通道
            self.fnirs_freq_bands1 = []
            self.fnirs_note = None

