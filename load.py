# 方法1: 如果文件是用torch.save保存的tensor
import torch


def inspect_edf_tensor(file_path):
    # 加载数据
    data = torch.load(file_path)

    print("数据类型:", type(data))
    print("数据维度:", data.shape)
    print("数据范围:", data.min().item(), "to", data.max().item())
    print("均值:", data.mean().item())
    print("标准差:", data.std().item())

    # 打印前几个通道的基本信息
    print("\n前3个通道的前10个采样点:")
    print(data[:3, :10])

    return data


# 方法2: 如果文件是标准的EDF格式
import mne


def inspect_edf_mne(file_path):
    # 加载数据
    raw = mne.io.read_raw_edf(file_path, preload=True)

    print("数据信息:")
    print(raw.info)
    print("\n通道名称:")
    print(raw.ch_names)
    print("\n采样率:", raw.info['sfreq'], "Hz")
    print("数据时长:", raw.times.max(), "秒")
    print("数据维度:", raw.get_data().shape)

    # 显示前几个通道的数据
    data = raw.get_data()
    print("\n前3个通道的前10个采样点:")
    print(data[:3, :10])

    return raw


# 使用示例:
file_path = "/data1/wangkuiyu/preprocess_code/EEGPT/pretrain_set_10bands/TrainFolder/0/eeg_45win_0.edf"

try:
    # 先尝试作为tensor加载
    print("=== 尝试作为Tensor加载 ===")
    data = inspect_edf_tensor(file_path)
    print('1111')
except:
    try:
        # 如果失败，尝试作为标准EDF加载
        print("\n=== 尝试作为EDF加载 ===")
        raw = inspect_edf_mne(file_path)
    except:
        print("无法加载文件，请检查文件格式")