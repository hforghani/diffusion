import os


def mkdir_rec(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.mkdir(dir_path)
        except FileNotFoundError:
            mkdir_rec(os.path.dirname(dir_path))
            os.mkdir(dir_path)
