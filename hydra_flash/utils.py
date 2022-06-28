from pathlib import Path


def resolver(root, file_id):
    return Path(root) / f"{file_id}.jpg"
