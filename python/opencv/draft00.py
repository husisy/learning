import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.ion()

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())


def generate_random_image(filepath=None):
    if filepath is None:
        filepath = hf_file('random{}.jpg'.format(np.random.randint(0, 100000, size=())))
    fig,ax = plt.subplots()
    for color in ['tab:blue', 'tab:orange', 'tab:green']:
        n = 750
        x, y = np.random.rand(2, n)
        scale = 200.0 * np.random.rand(n)
        ax.scatter(x, y, c=color, s=scale, label=color, alpha=0.3, edgecolors='none')
    fig.savefig(filepath)
    plt.close(fig)
    return filepath

filepath = generate_random_image()
z0 = cv2.imread(filepath) #(np,uint8,(H,W,3))
with open(filepath, 'rb') as fid:
    z1 = cv2.imdecode(np.frombuffer(fid.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

# plt.imshow(img)

# cv2.imwrite(hf_file('tbd00.jpg'), img)
