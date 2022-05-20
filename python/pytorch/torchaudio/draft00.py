# https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

hf_data = lambda *x: os.path.join('data', *x)
if not os.path.exists(hf_data()):
    os.makedirs(hf_data())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def isnotebook():
    # https://stackoverflow.com/q/33160744
    tmp0 = globals()
    if 'get_ipython' in tmp0:
        shell = tmp0['get_ipython'].__class__.__name__
        if shell == 'ZMQInteractiveShell': # Jupyter notebook or qtconsole
            ret = True
        elif shell == 'TerminalInteractiveShell': #Terminal running IPython
            ret = False
        else:
            ret = False
    else:
        ret = False
    return ret


def play_wav_audio(path):
    if isnotebook():
        import IPython
        ret = IPython.display.Audio(path)
    else:
        assert path.endswith('.wav')
        import platform
        if platform.system()=='Windows':
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        ret = None
    return ret


def get_sample_audio():
    url = ("https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/"
            "train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
    path = hf_data("speech.wav")
    if not os.path.exists(path):
        import requests
        with open(path, "wb") as file:
            file.write(requests.get(url).content)
    return path


def greedy_ctc_decode(emission:torch.tensor, label_list, blank:int=0)->str:
    indices = torch.argmax(emission, dim=-1)  # [num_seq,]
    indices = torch.unique_consecutive(indices, dim=-1)
    indices = [x for x in indices if (x!=blank)]
    ret = "".join([label_list[i] for i in indices])
    return ret


SPEECH_FILE = get_sample_audio()
# play_wav_audio(SPEECH_FILE)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
bundle.sample_rate #16000
bundle.get_labels() #<s> <pad> </s> <unk> | E T A O N I H S R D L U M W C F G Y P B V K ' X J Q Z
model = bundle.get_model().to(device)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
# waveform (torch,float32,(1,54400))
# sample_rate: 16000
waveform = waveform.to(device)
if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


with torch.inference_mode():
    features, _ = model.extract_features(waveform)
    # features (list,(torch,float32,(1,169,768),12)
    #   169: Frame (time-axis)
    #   768: Feature dimension

    emission, _ = model(waveform)
    # emission (torch,float32,(1,169,32))

# fig,ax = plt.subplots()
# ax.imshow(emission[0].cpu().T)
# ax.set_title("Classification result")
# ax.set_xlabel("Frame (time-axis)")
# ax.set_ylabel("Class")

transcript = greedy_ctc_decode(emission[0], label_list=bundle.get_labels())
