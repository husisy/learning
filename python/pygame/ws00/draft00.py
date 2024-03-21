import os
import subprocess
import shutil
import tempfile
import pygame


def download_asset_if_not_exist():
    git_url = 'git@github.com:clear-code-projects/UltimatePygameIntro.git'
    if not os.path.exists('audio'):
        with tempfile.TemporaryDirectory() as z0:
            subprocess.run(['git', 'clone', git_url, z0])
            for x in ['audio', 'font', 'graphics']:
                if os.path.exists(x):
                    shutil.rmtree(x)
                shutil.move(os.path.join(z0, x), x)


if __name__=='__main__':
    download_asset_if_not_exist()
