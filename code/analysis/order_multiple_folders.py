#%%
"""
Some folders contain multiple data acquisition image sets from separate multi-dimensional acquisition runs
on Micro-Manager. While the original creation of the dataset is preferred, during file transfer, the metadata
on computers may change the creation date and time to when the folder is downloaded. Instead, we insert a file
that contains information on the order that the files are created.
"""
import os
import pandas as pd
import numpy as np

root = '../../../data/active_stress/atpdilutions/'
directory_list = np.sort([directory for directory in os.listdir(root) if ('slide' in directory) and (len(str(directory))<25)])
#%%
for directory in directory_list:
    subdir_list = np.sort([os.path.join(root,str(directory),subdir) for subdir in os.listdir(os.path.join(root,str(directory))) if ('slide' in subdir) and ('.csv' not in subdir)])

    filename = []
    red_time = []
    yellow_time = []
    base_time = []

    df = pd.DataFrame([])
    for n in range(len(subdir_list)):
        filename.append(str(subdir_list[n]))

        fd = os.open(os.path.join(subdir_list[n],'metadata.txt'), os.O_RDONLY)
        bytes = 500000
        metadata = str(os.read(fd, bytes))
        idx_summary = [i for i in range(len(metadata)) if metadata.startswith('Summary', i)]

        summary_snippet = metadata[idx_summary[0]:idx_summary[1]]

        idx_chlists = [i for i in range(len(summary_snippet)) if summary_snippet.startswith('"ChNames"', i)][0]
        idx_chliste = [i for i in range(len(summary_snippet)) if summary_snippet.startswith('"IJType"', i)][0]

        idx_time = [i+20 for i in range(len(summary_snippet)) if summary_snippet.startswith('"Time": ', i)][0]
        idx_timeend = [i-11 for i in range(len(summary_snippet)) if summary_snippet.startswith('"Date":', i)][0]
        hrs = int(summary_snippet[idx_time:idx_time+2])
        min = int(summary_snippet[idx_time+3:idx_time+5])
        sec = int(summary_snippet[idx_time+6:idx_time+8])
        base_time.append(3600*hrs + 60*min + sec)

        channel_list = summary_snippet[idx_chlists:idx_chliste]

        if 'Red' not in channel_list:
            red_time.append(np.nan)
        elif 'Yellow' not in channel_list:
            yellow_time.append(np.nan)

        num_channels = 1 + len([i for i in range(len(channel_list)) if channel_list.startswith(',',i)])

        # Grab time stamp for first image acquisition
        for ch in range(num_channels):
            if ch == len(idx_summary)-1:
                summary = metadata[idx_summary[ch]:]
            elif ch >= len(idx_summary):
                continue
            else:
                summary = metadata[idx_summary[ch]:idx_summary[ch+1]]

            # Find where Channels are specified
            idx_channel = [i+12 for i in range(len(summary)) if summary.startswith('"Channel": ', i)]
            if len(idx_channel)==0:
                continue
            idx_core = [i for i in range(len(summary)) if summary.startswith('"Core-Initialize":', i)]
            channel = summary[idx_channel[0]:idx_core[0]-6]
            if ('All_Off' in channel) or ('Blue' in channel):
                continue

            idx_time = [i+20 for i in range(len(summary)) if summary.startswith('"Time": ', i)]
            idx_timeend = [i-11 for i in range(len(summary)) if summary.startswith('"Arduino-Switch-Blank On":', i)]

            hrs = int(summary[idx_time[1]:idx_time[1]+2])
            min = int(summary[idx_time[1]+3:idx_time[1]+5])
            sec = int(summary[idx_time[1]+6:idx_time[1]+8])
            time = 3600*hrs + 60*min + sec
            if 'Red' in channel:
                red_time.append(time)
            elif 'Yellow' in channel:
                yellow_time.append(time)
        if len(red_time) != len(yellow_time):
            while len(red_time) < len(yellow_time):
                red_time.append(np.nan)
            while len(yellow_time) < len(red_time):
                yellow_time.append(np.nan)

    df['filename'] = filename
    df['time (red)'] = red_time - np.nanmin(red_time)
    df['time (yellow)'] = yellow_time - np.nanmin(yellow_time)
    df['initial acquisition'] = base_time - np.nanmin(base_time)
    df = df.sort_values('initial acquisition')
    df['order'] = np.arange(0, len(subdir_list), 1)
    df.to_csv(os.path.join(root,str(directory),'%s_filename_order_atpncd236.csv' %str(directory)),sep=',')

# %%
