# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, File, Input, Path
import io
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plt_fig_to_nparray(fig):
    # https://stackoverflow.com/a/32908899
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    with Image.open(buffer) as image_pil:
        ret = np.array(image_pil)
    return ret

class Predictor(BasePredictor):
    def predict(self, text:str="hello world") -> File:
        fig,ax = plt.subplots()
        ax.text(0, 0, text, horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        fig.tight_layout()
        filename = str(random.randint(0, int(1e18))).rjust(18, '0') + '.png'
        fig.savefig(filename)
        # image_pil = Image.open(filename)
        # image_pil = Image.fromarray(plt_fig_to_nparray(fig))
        # File(image_pil) #fail for no reason
        return Path(filename)
