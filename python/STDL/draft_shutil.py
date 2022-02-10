import os
import shutil

def detect_gpu_available():
    path = shutil.which('nvidia-smi')
    ret = False
    if path is not None:
        ret = os.access(path, os.X_OK)
    return ret
