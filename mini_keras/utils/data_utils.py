import os

import requests


def get_file(url, filename, cache_dir_location="datasets"):
    # TODO: Add more parameters such as Untar, Extract and more features, so It's useful for the Enduser API.
    cache_location = os.path.join(
        os.path.expanduser("~"), ".mini_keras", cache_dir_location
    )

    if not os.path.exists(cache_location):
        os.makedirs(cache_location)

    exists = True

    full_path = os.path.join(cache_location, filename)

    if not os.path.exists(full_path):
        exists = False

    if not exists:
        try:
            req = requests.get(url, allow_redirects=True)

            with open(full_path, "wb") as file:
                for chunk in req.iter_content(chunk_size=4096):
                    file.write(chunk)
        except Exception as exc:
            if os.path.exists(full_path):
                os.remove(full_path)

            raise exc

    return full_path
