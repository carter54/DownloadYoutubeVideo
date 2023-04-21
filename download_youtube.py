"""
This file downloads youtube videos providing the url of each video.

1. It downloads the best quality video and audio, save the video ends with .mp4.mkv
2. It crop video from 00:00:10 to 00:10:10, maximum 10min in total.
3. It changes video fps to 25

Usage:
```
CUDA_VISIBLE_DEVICES=0 python download_youtube.py \
--source_dir youtube_link/url.txt \
--output_dir /home/hr/dataset/youtube/ \
--num_workers 4
```
"""


import os
import argparse
from typing import List, Dict
from multiprocessing import Pool
import subprocess
import numpy as np
import librosa
import soundfile as sf
from subprocess import Popen, PIPE
from urllib import parse

from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

subsets = ["RD", "WDA", "WRA"]


def process_file(model, audio_file_name, out_file_name):
    '''
    Funtion to read an audio file, rocess it by the network and write the
    enhanced audio to .wav file.

    Parameters
    ----------
    model : Keras model
        Keras model, which accepts audio in the size (1,timesteps).
    audio_file_name : STRING
        Name and path of the input audio file.
    out_file_name : STRING
        Name and path of the target file.

    '''

    # read audio file with librosa to handle resampling and enforce mono
    in_data, fs = librosa.core.load(audio_file_name, sr=16000, mono=True)
    # get length of file
    len_orig = len(in_data)
    # pad audio
    zero_pad = np.zeros(384)
    in_data = np.concatenate((zero_pad, in_data, zero_pad), axis=0)
    # predict audio with the model
    predicted = model.predict_on_batch(
        np.expand_dims(in_data, axis=0).astype(np.float32))
    # squeeze the batch dimension away
    predicted_speech = np.squeeze(predicted)
    predicted_speech = predicted_speech[384:384 + len_orig]
    # write the file to target destination
    sf.write(out_file_name, predicted_speech, fs)


def download_hdtf(source_dir: os.PathLike, output_dir: os.PathLike, num_workers: int, **process_video_kwargs):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '_videos_raw'), exist_ok=True)

    download_queue = construct_download_queue(source_dir, output_dir)
    task_kwargs = [dict(
        video_data=vd,
        output_dir=output_dir,
        **process_video_kwargs,
     ) for vd in download_queue]
    pool = Pool(processes=num_workers)
    tqdm_kwargs = dict(total=len(task_kwargs), desc=f'Downloading videos into {output_dir} ')

    for _ in tqdm(pool.imap_unordered(task_proxy, task_kwargs), **tqdm_kwargs):
        pass

    print('Download is finished, you can now (optionally) delete the following directories, since they are not needed anymore and occupy a lot of space:')
    print(' -', os.path.join(output_dir, '_videos_raw'))


def construct_download_queue(source_dir: os.PathLike, output_dir: os.PathLike) -> List[Dict]:
    download_queue = []

    video_urls = read_file_as_space_separated_data(source_dir)

    for video_name, (video_url,) in video_urls.items():
        download_queue.append({
            'name': f'{video_name}',
            'id': video_name,
            'output_dir': output_dir,
            'url': video_url
        })

    return download_queue


def task_proxy(kwargs):
    return download_and_process_video(**kwargs)


def download_and_process_video(video_data: Dict, output_dir: str):
    """
    Downloads the video and cuts/crops it into several ones according to the provided time intervals
    """
    raw_download_path = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}.mp4")
    raw_download_log_file = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}_download_log.txt")
    download_result = download_video(video_data['url'], raw_download_path, log_file=raw_download_log_file)

    if not download_result:
        print('Failed to download', video_data)
        print(f'See {raw_download_log_file} for details')
        return

    input_path = raw_download_path + '.mkv'
    crop_success = cut_10min_25fps(input_path, raw_download_path, "00:00:00", "00:30:00", "25")
    if not crop_success:
        print(f'Failed to cut_10min_25fps ', video_data)
    else:
        os.remove(input_path)


def read_file_as_space_separated_data(filepath: os.PathLike) -> Dict:
    """
    Reads a file as a space-separated dataframe, where the first column is the index
    """
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        lines = [[v.strip() for v in l.strip().split(' ')] for l in lines]
        data = {i: l[1:] for i, l in enumerate(lines)}

    return data


def download_video(url, download_path, video_format="mp4", log_file=None):
    """
    Download video from YouTube.
    :param video_id:        YouTube ID of the video.
    :param download_path:   Where to save the video.
    :param video_format:    Format to download.
    :param log_file:        Path to a log file for youtube-dl.
    :return:                Tuple: path to the downloaded video and a bool indicating success.

    Copy-pasted from https://github.com/ytdl-org/youtube-dl
    """
    # if os.path.isfile(download_path): return True # File already exists
    if log_file is None:
        stderr = subprocess.DEVNULL
    else:
        stderr = open(log_file, "a")
    video_selection = f"bestvideo[ext={video_format}]+bestaudio"
    video_selection = video_selection
    command = [
        "yt-dlp",
        "--no-check-certificate",
        "--prefer-insecure",
        url, "--quiet", "-f",
        video_selection,
        "--output", download_path,
        "--no-continue",
    ]
    return_code = subprocess.call(command, stderr=stderr)
    success = return_code == 0

    if log_file is not None:
        stderr.close()

    return success and os.path.isfile(download_path+'.mkv')


def cut_10min_25fps(raw_video_path, output_path, start, end, fps):
    command = ' '.join([
        "ffmpeg", "-i", raw_video_path,
        "-loglevel", "quiet", # Verbosity arguments
        "-y", # Overwrite if the file exists
        "-async", "1",
        "-ss", str(start), "-t", str(end), # Cut arguments
        "-r", str(fps),
        output_path
    ])

    return_code = subprocess.call(command, shell=True)
    success = return_code == 0

    if not success:
        print('Command failed:', command)

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download youtube dataset")
    parser.add_argument('-s', '--source_dir', type=str, default="/home/hr/PycharmProjects/video_download_and_process/youtube_link/added_url_20230419.txt", help='Path to the directory with the dataset')
    parser.add_argument('-o', '--output_dir', type=str, default="/opt/data/youtube_20230419/", help='Where to save the videos?')
    parser.add_argument('-w', '--num_workers', type=int, default=8, help='Number of workers for downloading')
    args = parser.parse_args()

    download_hdtf(
        args.source_dir,
        args.output_dir,
        args.num_workers,
    )
