import os

import numpy as np
import requests


def get_file(url, filepath, cache=True):
    file = requests.get(url)
    file_ext = url.split("/")[-1]
    if cache is True and os.path.exists(f"{os.path.join(filepath, file_ext)}"):
        print("The file exists")
    else:
        open(f"{os.path.join(filepath, file_ext)}", 'wb').write(file.content)
