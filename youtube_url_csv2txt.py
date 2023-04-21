import pandas as pd
import os

csv_path = 'youtube_link/added_csv_20230419'
txt_path = 'youtube_link/added_url_20230419.txt'

for csv_file in os.listdir(csv_path):
    content = pd.read_csv(os.path.join(csv_path, csv_file))
    columns = ['', 'web', '', 'url']
    content.columns = columns

    content[['web', 'url']][:50].to_csv(txt_path, header=None, index=None, sep=' ', mode='a')
