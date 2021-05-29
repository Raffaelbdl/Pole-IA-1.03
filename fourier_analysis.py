"""
Performing a Fourier transform on raw data in order to get a frequency analysis of the signals.
"""

from utils.io import readFile, MergeBodyAcc, MergeBodyGyro
import numpy as np
from scipy.fftpack import fft
import seaborn as sn
import matplotlib.pyplot as plt


def analysis_subplot(array, start_line, n, Dt, n_col, rows, cols, i_subplot, x_label, y_label, title):
    time_array = np.linspace(0, n*Dt, n*n_col)
    data = array[start_line:start_line+n].reshape(n*n_col)
    plt.subplot(rows, cols, i_subplot)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(time_array, data)
    return data


def draw_fft_subplot(data, n, Dt, n_col, rows, cols, i_subplot, x_label, y_label, title):
    f_ech = n_col/Dt
    freq_array = np.linspace(0, f_ech/2, n*n_col//2)
    data_freq = fft(data)[:n*n_col//2]
    plt.subplot(rows, cols, i_subplot)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(freq_array, 2*np.abs(data_freq)/n/n_col)
    return data_freq


def time_and_freq_analysis_plots(array, y_label, nb_timelapse, Dt, n_col):
    print("########### Time domain data visualization ###########")
    print("Plotting graphics ....................................")
    plt.suptitle("Time domain data visualization of "+y_label)
    n = nb_timelapse
    start = 0
    standing_data = analysis_subplot(
        array, start, n, Dt, n_col, 2, 2, 1, "time (s)", y_label, "When standing")

    start = 32
    sitting_data = analysis_subplot(
        array, start, n, Dt, n_col, 2, 2, 2, "time (s)", y_label, "When sitting")

    start = 80
    walking_data = analysis_subplot(
        array, start, n, Dt, n_col, 2, 2, 3, "time (s)", y_label, "When walking")

    start = 110
    walking_downstairs_data = analysis_subplot(array, start, n, Dt, n_col, 2, 2, 4,
                                               "time (s)", y_label, "When walking downstairs")
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    print("########### Freq domain data visualization ###########")
    print("Plotting graphics ....................................")

    plt.suptitle("Freq domain data visualization of "+y_label)

    standing_data_freq = draw_fft_subplot(
        standing_data, n, Dt, n_col, 2, 2, 1, "frequency (Hz)", y_label, "When standing")

    sitting_data_freq = draw_fft_subplot(
        sitting_data, n, Dt, n_col, 2, 2, 2, "frequency (Hz)", y_label, "When sitting")

    walking_data_freq = draw_fft_subplot(
        walking_data, n, Dt, n_col, 2, 2, 3, "frequency (Hz)", y_label, "When walking")

    walking_downstairs_data_freq = draw_fft_subplot(walking_downstairs_data, n, Dt, n_col, 2, 2, 4,
                                                    "frequency (Hz)", y_label, "When walking downstairs")
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


if __name__ == '__main__':
    sn.set()

    # Analyse des signaux temporels

    body_acc = MergeBodyAcc()
    body_gyro = MergeBodyGyro()

    n_row, n_col, _ = body_acc.shape

    Dt = 2.56  # s

    n = 5   # nb plages

    main_arrays = [body_acc, body_gyro]
    y_labels = ["body_{}_{} ({})".format(key, axis, unit)
                for key, unit in [("acc", "g"), ("gyro", "rd/s")] for axis in ["x", "y", "z"]]
    params = [{"array": main_arrays[i//3][:, :, i %
                                          3], "y_label":y_labels[i]} for i in range(6)]

    for d in params:
        array = d["array"]
        y_label = d["y_label"]
        time_and_freq_analysis_plots(array, y_label, n, Dt, n_col)
