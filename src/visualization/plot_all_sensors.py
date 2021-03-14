"""
plot_all_sensors.py
For a user id, plot all the sensor data in a figure
"""

import matplotlib.pylab as plt
import librosa
import librosa.display
import pandas as pa
import sqlite3
import numpy as np
import config

from src.data import load_weight_sensor, load_water_distance, load_radar

import warnings
warnings.filterwarnings("ignore")


def PlotUse2(
    user_id: int,
    fig_path: Option[str] = config.SAVE_PLOT_ALL_SENSORS_PATH
) -> None:
    """
    Plot all sensors data into one figure for user id.

    :param user_id: an integer for user id
    :param fig_path: a string of location of saved plot
    """
    data_d = {}
    thermal_mi = GetThermal2(use_i)  # thermal camera

    seat, foot = load_weight_sensor.get_seat_and_foot_weight_clean(user_id)
    water_distance = load_water_distance.get_water_distance_clean(user_id)
    radar_sum = load_radar.get_radar_sum_clean(user_id)

    if use_i < 945 or use_i >= 1755:
        fig, (ax1, ax2, ax3, ax4, ax7, ax8, ax9, ax10, ax11,
              ax12) = plt.subplots(10, sharex=True, figsize=(30, 20))
    else:  # for now, just assume we're only looking at samples that have audio files.
        fig, (ax1, ax2, ax3, ax4, ax7, ax8, ax9, ax10, ax11, ax12,
              ax13) = plt.subplots(11, sharex=True, figsize=(30, 20))

    fig.tight_layout(pad=3.0)

    # seat scale
    ax1.plot(seat, linewidth=3)
    ax1.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    ax1.set_ylim(seat.median()-1, seat.median()+1)
    ax1.set_ylabel('seat [kg]')
    ax1.grid()
    ax1.tick_params(axis='x', labelbottom=True, labelrotation=90)

    # foot scale
    ax2.plot(foot, linewidth=3)
    ax2.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    ax2.set_ylim(foot.median()-1, foot.median()+1)
    ax2.tick_params(axis='x', labelbottom=True, labelrotation=90)
    ax2.set_ylabel('footrest [kg]')
    ax2.grid()

    # total weight scale
    ax3.plot(x_ix, sumScale_sz, linewidth=3)
    ax3.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    ax3.set_ylim(sumScale_sz.median()-1, sumScale_sz.median()+1)
    ax3.tick_params(axis='x', labelbottom=True, labelrotation=90)
    ax3.set_ylabel('sum_scale [kg]')
    ax3.grid()

    # water distance
    ax4.plot(x_ix, data_sz, linewidth=3)
    ax4.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)
    axi.set_ylabel('water distance [cm]')
    ax4.grid()

    #


def PlotUse(
    user_id: int,
    fig_path: Option[str] = config.SAVE_PLOT_ALL_SENSORS_PATH
) -> None:
    data_d = {}
    data_d[1] = GetSensor(use_i, 1)  # vertical ultrasound
    data_d[2] = GetSensor(use_i, 2)  # seat scale
    data_d[3] = GetSensor(use_i, 3)  # foot scale
    thermal_mi = GetThermal2(use_i)  # thermal camera
    radar_df = GetRadar(use_i)  # radar

    t0 = data_d[1][0][0]
    data_d['urineButton'] = GetEvent(use_i, 2)
    data_d['stoolButton'] = GetEvent(use_i, 3)
    stool_l = [float(i-t0)/1000 for i in data_d['stoolButton']]
    urine_l = [float(i-t0)/1000 for i in data_d['urineButton']]

    event_start_t = np.inf
    event_end_t = 0.
    if len(stool_l) > 0:
        event_start_t = min(event_start_t, min(stool_l))
        event_end_t = max(event_end_t, max(stool_l))
    # endif

    if len(urine_l) > 0:
        event_start_t = min(event_start_t, min(urine_l))
        event_end_t = max(event_end_t, max(urine_l))
    # endif

    clean1_sz, clean2_sz = cleanSensors(
        data_d[2][0], data_d[2][1], data_d[3][0], data_d[3][1])

    if use_i < 945:
        fig, (ax1, ax2, ax3, ax4, ax7, ax8, ax9, ax10, ax11,
              ax12) = plt.subplots(10, sharex=True, figsize=(30, 20))
    else:  # for now, just assume we're only looking at samples that have audio files.
        fig, (ax1, ax2, ax3, ax4, ax7, ax8, ax9, ax10, ax11, ax12, ax13,
              ax14) = plt.subplots(12, sharex=True, figsize=(30, 20))
    # endif

    fig.tight_layout(pad=3.0)

    footSeat_space_n = 4

    axi = ax1
    seatScale_sz = clean1_sz/1000
    x_ix = (seatScale_sz.index-t0)/1000
    axi.plot(x_ix, seatScale_sz, linewidth=3)
    axi.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)

    axi.set_ylabel('seat [kg]')
    axi.grid()
    event_b = (x_ix > event_start_t) & (x_ix < event_end_t)
    event_scale_sz = seatScale_sz[event_b]
    axi.set_ylim(seatScale_sz.median()-footSeat_space_n,
                 seatScale_sz.median()+footSeat_space_n)
    # axi.set_ylim(30,90)

    for i in range(int(len(stool_l)/2)):
        x = [stool_l[2*i], stool_l[2*i], stool_l[2*i+1], stool_l[2*i+1]]
        y = [0, 500, 500, 0]
        axi.fill(x, y, 'red', alpha=0.5)
    # endfor
    for i in range(int(len(urine_l)/2)):
        x = [urine_l[2*i], urine_l[2*i], urine_l[2*i+1], urine_l[2*i+1]]
        y = [0, 500, 500, 0]
        axi.fill(x, y, 'gold', alpha=0.8)
    # endfor

    axi = ax2
    footScale_sz = clean2_sz/1000

    x_ix = (footScale_sz.index-t0)/1000
    axi.plot(x_ix, footScale_sz, linewidth=3)
    axi.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)
    axi.set_ylabel('footrest [kg]')
    axi.grid()
    event_b = (x_ix > event_start_t) & (x_ix < event_end_t)

    event_scale_sz = footScale_sz[event_b]
    axi.set_ylim(footScale_sz.median()-footSeat_space_n,
                 footScale_sz.median()+footSeat_space_n)
    # axi.set_ylim(0,90)

    for i in range(int(len(stool_l)/2)):
        x = [stool_l[2*i], stool_l[2*i], stool_l[2*i+1], stool_l[2*i+1]]
        y = [0, 500, 500, 0]
        axi.fill(x, y, 'red', alpha=0.5)
    # endfor
    for i in range(int(len(urine_l)/2)):
        x = [urine_l[2*i], urine_l[2*i], urine_l[2*i+1], urine_l[2*i+1]]
        y = [0, 500, 500, 0]
        axi.fill(x, y, 'gold', alpha=0.8)
    # endfor

    axi = ax3
    sumScale_sz = seatScale_sz + footScale_sz

    x_ix = (sumScale_sz.index-t0)/1000
    axi.plot(x_ix, sumScale_sz, linewidth=3)
    axi.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)
    axi.set_ylabel('sum_scale [kg]')
    event_b = (x_ix > event_start_t) & (x_ix < event_end_t)

    event_scale_sz = sumScale_sz[event_b]
    axi.set_ylim(sumScale_sz.median()-3, sumScale_sz.median()+3)
    # axi.set_ylim(75,95)

    axi.grid()
    for i in range(int(len(stool_l)/2)):
        x = [stool_l[2*i], stool_l[2*i], stool_l[2*i+1], stool_l[2*i+1]]
        y = [0, 500, 500, 0]
        axi.fill(x, y, 'red', alpha=0.5)
    # endfor
    for i in range(int(len(urine_l)/2)):
        x = [urine_l[2*i], urine_l[2*i], urine_l[2*i+1], urine_l[2*i+1]]
        y = [0, 500, 500, 0]
        axi.fill(x, y, 'gold', alpha=0.8)
    # endfor

    axi = ax4
    data_sz = pa.Series([i*100 for i in data_d[1][1]])

    x_ix = [(i-t0)/1000 for i in data_d[1][0]]
    axi.plot(x_ix, data_sz, linewidth=3)
    axi.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)
    axi.set_ylabel('water distance [cm]')
    axi.grid()

    axi.set_ylim(data_sz.median()-1, data_sz.median()+1)
    for i in range(int(len(stool_l)/2)):
        x = [stool_l[2*i], stool_l[2*i], stool_l[2*i+1], stool_l[2*i+1]]
        y = [0, 20, 20, 0]
        axi.fill(x, y, 'red', alpha=0.5)
    # endfor
    for i in range(int(len(urine_l)/2)):
        x = [urine_l[2*i], urine_l[2*i], urine_l[2*i+1], urine_l[2*i+1]]
        y = [0, 20, 20, 0]
        axi.fill(x, y, 'gold', alpha=0.8)
    # endfor
    x_ax_lim = axi.get_xlim()

    axi = ax7
    t0_df = thermal_mi.loc[thermal_mi.index[0][0]]
    diff_d = {}
    for t, df in thermal_mi.groupby(level=0):
        diff_d[t] = ((t0_df - thermal_mi.loc[t])**2).sum().sum()
    # endfor

    diff_sz = pa.Series(diff_d)
    x_ix = (diff_sz.index-t0)/1000
    axi.plot(x_ix, diff_sz, linewidth=3)
    axi.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)
    axi.set_ylabel('infrared camera')

    axi.grid()
    for i in range(int(len(stool_l)/2)):
        x = [stool_l[2*i], stool_l[2*i], stool_l[2*i+1], stool_l[2*i+1]]
        y = [0, 8000, 8000, 0]
        axi.fill(x, y, 'red', alpha=0.5)
    # endfor
    for i in range(int(len(urine_l)/2)):
        x = [urine_l[2*i], urine_l[2*i], urine_l[2*i+1], urine_l[2*i+1]]
        y = [0, 8000, 8000, 0]
        axi.fill(x, y, 'gold', alpha=0.8)
    # endfor
    axi.set_xlim(x_ax_lim)

    # plot thermal camera data
    axi = ax8
    row_df = thermal_mi.groupby(level=0).mean()

    xval = (row_df.index - t0)/1000
    yval = range(8)[::-1]
    xx, yy = np.meshgrid(xval, yval)
    axi.pcolormesh(xx, yy, row_df.T, edgecolors='face')
    axi.set_ylabel('row_infrared')
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)

    axi = ax9
    column_df = thermal_mi.mean(1).unstack(level=1)
    xval = (column_df.index - t0)/1000
    yval = range(8)[::-1]
    xx, yy = np.meshgrid(xval, yval)
    axi.pcolormesh(xx, yy, column_df.T, edgecolors='face')
    axi.set_ylabel('column_infrared')
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)

    # plot radar data
    floor_i = 50
    ceil_i = 200
    axi = ax10
    x0 = (radar_df.columns[0]-t0)/1000
    x1 = (radar_df.columns[-1]-t0)/1000
    axi.imshow(radar_df.iloc[::-1], aspect='auto',
               extent=[x0, x1, radar_df.index[0], radar_df.index[-1]])
    axi.plot([x0, x1], [floor_i, floor_i], 'r-')
    axi.plot([x0, x1], [ceil_i, ceil_i], 'r-')
    axi.set_ylabel('radar')
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)

    axi = ax11
    area_d = {}
    for i in radar_df.columns:
        sq_sz = (radar_df[i])**2
        area_d[i] = sum(sq_sz.iloc[floor_i:ceil_i])
    # endfor
    area_sz = pa.Series(area_d)
    x_ix = (area_sz.index-t0)/1000
    axi.plot(x_ix, area_sz, linewidth=3)
    axi.tick_params(axis='x', labelbottom=True, labelrotation=90)
    axi.set_ylabel('radar sum')
    axi.set_ylim([0, 5e9])
    axi.grid()
    for i in range(int(len(stool_l)/2)):
        x = [stool_l[2*i], stool_l[2*i], stool_l[2*i+1], stool_l[2*i+1]]
        y = [0, 5e9, 5e9, 0]
        axi.fill(x, y, 'red', alpha=0.5)
    # endfor
    for i in range(int(len(urine_l)/2)):
        x = [urine_l[2*i], urine_l[2*i], urine_l[2*i+1], urine_l[2*i+1]]
        y = [0, 5e9, 5e9, 0]
        axi.fill(x, y, 'gold', alpha=0.8)
    # endfor

    if use_i < 945:
        axi = ax12
        S_DB, fs, hop_len = GetAudio1(use_i)
        ax = librosa.display.specshow(
            S_DB, sr=fs, hop_length=hop_len, x_axis='time', y_axis='mel', ax=axi)
        axi.set_ylabel('back mic')
    elif use_i < 1753:
        axi = ax12
        S_DB, fs, hop_len = GetAudio2(use_i, "back")
        ax = librosa.display.specshow(
            S_DB, sr=fs, hop_length=hop_len, x_axis='time', y_axis='mel', ax=axi)
        axi.set_ylabel('back mic')

        axi = ax13
        S_DB, fs, hop_len = GetAudio2(use_i, "front")
        ax = librosa.display.specshow(
            S_DB, sr=fs, hop_length=hop_len, x_axis='time', y_axis='mel', ax=axi)
        axi.set_ylabel('front mic')
    else:
        axi = ax12
        S_DB_front, S_DB_back, fs, hop_len = GetAudio3(use_i)
        ax = librosa.display.specshow(
            S_DB_front, sr=fs, hop_length=hop_len, x_axis='time', y_axis='mel', ax=axi)
        axi.set_ylabel('front mic')
        axi.tick_params(axis='x', labelbottom=True, labelrotation=90)

        axi = ax13
        ax = librosa.display.specshow(
            S_DB_back, sr=fs, hop_length=hop_len, x_axis='time', y_axis='mel', ax=axi)
        axi.set_ylabel('back mic')
        axi.tick_params(axis='x', labelbottom=True, labelrotation=90)


def GetAudio2(use_i, pos_s):
    sampleRate_n = 44100
    #wav_fn  = 'data/data_frames/data_capture_{}/'.format(use_i) + pos_s + '_audio_data.wav'
    wav_fn = os.path.join(
        DATA_PATH, 'data_frames/data_capture_{}/'.format(use_i) + pos_s + '_audio_data.wav')
    x, fs = librosa.load(wav_fn, sr=sampleRate_n)

    n_fft = 2048
    hop_length = 512
    n_mels = 128
    S = librosa.feature.melspectrogram(x, sr=sampleRate_n, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB, fs, hop_length


def GetAudio1(use_i):
    sampleRate_n = 44100
    #wav_fn  = 'data/data_frames/data_capture_{}/audio_data.wav'.format(use_i)
    wav_fn = os.path.join(
        DATA_PATH, 'data_frames/data_capture_{}/audio_data.wav'.format(use_i))
    x, fs = librosa.load(wav_fn, sr=sampleRate_n)

    n_fft = 2048
    hop_length = 512
    n_mels = 128
    S = librosa.feature.melspectrogram(x, sr=sampleRate_n, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB, fs, hop_length


def GetRadar(use_i):

    #data_fn = 'data/data_frames/data_capture_{}/radar_data.txt'.format(use_i)
    data_fn = os.path.join(
        DATA_PATH, "data_frames/data_capture_{}/radar_data.txt".format(use_i))
    data_f = open(data_fn, 'rt')
    line_s = data_f.read()
    data_l = eval(line_s)

    # save array of images
    t0_sz = pa.Series(data_l[0]['data'])
    data_d = {}
    for j in range(len(data_l)):
        t = data_l[j]['timestamp_ms']
        j_sz = pa.Series(data_l[j]['data'][0])
        data_d[t] = j_sz
    # endfor
    data_df = pa.DataFrame(data_d)
    return data_df


def GetThermal2(use_i):

    # read in image text file
    #thermal_fn = 'data/data_frames/data_capture_{}/thermal_data.txt'.format(use_i)
    thermal_fn = os.path.join(
        DATA_PATH, "data_frames/data_capture_{}/thermal_data.txt".format(use_i))
    thermal_f = open(thermal_fn, 'rt')
    line_s = thermal_f.read()
    thermal_l = eval(line_s)

    # save array of images
    t0_sz = pa.Series(thermal_l[0]['data'])
    thermal_d = {}
    for j in range(len(thermal_l)):
        t = thermal_l[j]['timestamp_ms']
        j_df = pa.DataFrame(
            pa.Series(thermal_l[j]['data']).values.reshape(8, 8).T)
        thermal_d[t] = j_df
    # endfor
    thermal_mi = pa.concat(thermal_d.values(), keys=thermal_d.keys())

    return thermal_mi


def cleanSensors(sensor1_t_l, sensor1_y_l, sensor2_t_l, sensor2_y_l):
    min_t = min(min(sensor1_t_l), min(sensor2_t_l))
    max_t = max(max(sensor1_t_l), max(sensor2_t_l))

    # setup partitions
    step_t = 500
    min_floor_t = int(np.floor(min_t/step_t)*step_t)
    max_ceil_t = int(np.ceil(max_t/step_t)*step_t)

    step1_d = {}
    step2_d = {}
    for i in range(min_floor_t, max_ceil_t+step_t, step_t):
        step1_d[i] = []
        step2_d[i] = []
    # endfor

    # step through both and assign values to each partition
    for i in range(len(sensor1_t_l)):
        interval_t = int(np.floor(sensor1_t_l[i]/step_t)*step_t)
        step1_d[interval_t].append(sensor1_y_l[i])
    # endfor
    for i in range(len(sensor2_t_l)):
        interval_t = int(np.floor(sensor2_t_l[i]/step_t)*step_t)
        step2_d[interval_t].append(sensor2_y_l[i])
    # endfor

    # step through each partition and either take averages or set to nan
    clean1_d = {}
    for i in step1_d.keys():
        clean1_d[i] = np.mean(step1_d[i])
    # endfor
    clean1_sz = pa.Series(clean1_d)

    clean2_d = {}
    for i in step2_d.keys():
        clean2_d[i] = np.mean(step2_d[i])
    # endfor
    clean2_sz = pa.Series(clean2_d)

    return clean1_sz, clean2_sz


def GetSensor(use_i, sensor_i):
    sql_s = "SELECT timestamp_ms, value FROM data WHERE data_capture_id={} AND sensor_id={}".format(
        use_i, sensor_i)
    conn = sqlite3.connect(os.path.join(DATA_PATH, "toilet.db"))
    cursor = conn.execute(sql_s)
    time_measurements = []
    distance_measurements = []
    for row in cursor:
        time_measurements.append(row[0])
        distance_measurements.append(row[1])
    data_t = (time_measurements, distance_measurements)
    return data_t


def GetEvent(use_i, sensor_i):
    # was there a urination event?
    conn = sqlite3.connect(os.path.join(DATA_PATH, "toilet.db"))
    sql_s = "SELECT timestamp_ms FROM data_start_stop_time WHERE data_capture_id={} AND type_id={}".format(
        use_i, sensor_i)
    cursor = conn.execute(sql_s)
    time_l = []
    for row in cursor:
        time_l.append(row[0])
    return time_l


if __name__ == "__main__":
    data_captures_id = [2040]

    for data_capture in data_captures_id:
        print("printing {}".format(data_capture))
        PlotUse(data_capture, FIG_PATH)
