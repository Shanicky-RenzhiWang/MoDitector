import pickle
import json
import os
from datetime import datetime
def save_obj(data, datafile):
    with open(datafile, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_obj(datafile):
    with open(datafile, 'rb') as f:
        return pickle.load(f)
    
    
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_event_from_log(filename, event_describe='collision'):
    events = []
    with open(filename,'r') as f:
        for line in f:
            if 'event from the simulator' in line:
                if event_describe in line:
                    event_time = re.search(r'@\[([0-9]+)\]', line).group(1)
                    events.append(int(event_time))
    return events

def get_most_recent_folder(directory):
    # 获取指定目录下所有的文件夹
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # 如果没有文件夹，返回None
    if not folders:
        return None
    
    # 初始化最近的文件夹和最新的修改时间
    most_recent_folder = None
    most_recent_time = None
    
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        # 获取文件夹的修改时间戳
        folder_mtime = os.path.getmtime(folder_path)
        folder_time = datetime.fromtimestamp(folder_mtime)
        
        # 更新最近的文件夹
        if most_recent_time is None or folder_mtime > most_recent_time:
            most_recent_time = folder_mtime
            most_recent_folder = folder
    
    return most_recent_folder