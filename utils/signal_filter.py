"""
信号滤波工具

提供均值滤波和巴特沃斯低通滤波两种方法，用于 TE 过程信号的平滑去噪。

使用示例:
    from utils.signal_filter import mean_filter, lowpass_filter
    smoothed = mean_filter(raw_signal, window_size=5)
    filtered = lowpass_filter(raw_signal, cutoff_freq=10.0, sampling_rate=100.0)

命令行用法:
    python -m utils.signal_filter <文件夹路径> <CSV列名> [--window 5] [--cutoff 10.0] [--fs 100.0]
"""
import os
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve, butter, filtfilt


def mean_filter(signal, window_size):
    """均值滤波 (滑动窗口平均)"""
    return convolve(signal, np.ones(window_size) / window_size, mode='same')


def lowpass_filter(signal, cutoff_freq, sampling_rate):
    """巴特沃斯低通滤波"""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def process_folder(folder_path, window_size, cutoff_frequency, sampling_rate, column_name):
    """从文件夹中读取所有 CSV，合并信号后应用滤波"""
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    combined_signal = np.array([])

    for file_path in file_paths:
        data = pd.read_csv(file_path)
        if column_name not in data.columns:
            raise KeyError(f"列 '{column_name}' 不存在于文件 {file_path} 中。可用列: {list(data.columns)}")
        signal = data[column_name].values
        combined_signal = np.concatenate([combined_signal, signal])

    smoothed_signal = mean_filter(combined_signal, window_size)
    lowpass_filtered_signal = lowpass_filter(combined_signal, cutoff_frequency, sampling_rate)

    return combined_signal, smoothed_signal, lowpass_filtered_signal


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对 CSV 信号数据应用滤波器")
    parser.add_argument("folder", help="包含 CSV 文件的文件夹路径")
    parser.add_argument("column", help="CSV 文件中的信号列名称")
    parser.add_argument("--window", type=int, default=5, help="均值滤波窗口大小 (默认: 5)")
    parser.add_argument("--cutoff", type=float, default=10.0, help="低通滤波截止频率 Hz (默认: 10.0)")
    parser.add_argument("--fs", type=float, default=100.0, help="采样率 Hz (默认: 100.0)")
    args = parser.parse_args()

    original, smoothed, lowpass_filtered = process_folder(
        args.folder, args.window, args.cutoff, args.fs, args.column
    )

    plt.figure(figsize=(10, 6))
    plt.plot(original, label='Original Signal', alpha=0.5)
    plt.plot(smoothed, label=f'Mean Filter (Window={args.window})', alpha=0.5)
    plt.plot(lowpass_filtered, label=f'Lowpass Filter (Cutoff={args.cutoff}Hz)', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Original and Filtered Signals')
    plt.legend()
    plt.show()
