import os
import requests
import numpy as np

def get_file(url, filepath, cache=True):
    file = requests.get(url)
    file_ext = url.split("/") [-1]
    if cache == True and os.path.exists(f"{os.path.join(filepath, file_ext)}"):
        print("The file exists")
    else:
        open(f"{os.path.join(filepath, file_ext)}", 'wb').write(file.content)
