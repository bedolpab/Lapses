import os


def make_directory(folder_name: str):
    if not os.path.exists(f'{folder_name}'):
        os.makedirs(folder_name)
