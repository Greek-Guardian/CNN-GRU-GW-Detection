import numpy as np




def judge(series):
    series = series[1:(len(series)-4)]
    is_BBH_signal = 0
    mean = np.mean(series)
    std = np.std(series)
    fft = np.abs(np.fft(series))

    if abs(mean)<=0.015:
        is_BBH_signal = is_BBH_signal + 0.25
    elif abs(mean)<=0.1:
        is_BBH_signal = is_BBH_signal + 0.20
    elif abs(mean-1)<=0.01:
        is_BBH_signal = is_BBH_signal - 0.25
    elif abs(mean-1)<=0.1:
        is_BBH_signal = is_BBH_signal - 0.2

    if std>=0.3:
        is_BBH_signal = is_BBH_signal + 0.25
    elif std>=0.25:
        is_BBH_signal = is_BBH_signal + 0.20
    elif std<=0.015:
        is_BBH_signal = is_BBH_signal - 0.25
    elif std<=0.03:
        is_BBH_signal = is_BBH_signal - 0.2

    if fft[5]>=5:
        is_BBH_signal = is_BBH_signal + 0.25
    elif fft[5]>=1:
        is_BBH_signal = is_BBH_signal + 0.20
    elif fft[5]<=0.5:
        is_BBH_signal = is_BBH_signal - 0.25
    elif fft[5]<=0.2:
        is_BBH_signal = is_BBH_signal - 0.2

    if fft[-1]>=5:
        is_BBH_signal = is_BBH_signal + 0.25
    elif fft[-1]>=1:
        is_BBH_signal = is_BBH_signal + 0.20
    elif fft[-1]<=0.5:
        is_BBH_signal = is_BBH_signal - 0.25
    elif fft[-1]<=0.2:
        is_BBH_signal = is_BBH_signal - 0.2

    is_BBH_signal = (is_BBH_signal + 1)/2
    
    return is_BBH_signal