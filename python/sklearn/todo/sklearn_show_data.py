import numpy as np
import matplotlib.pyplot as plt


def show_digits():
    digits = datasets.load_digits()
    _, ax = plt.subplots(1, 4, figsize=(6.4,2.4))
    ind0 = np.random.permutation(digits.images.shape[0])[:4]
    for ind1,image,label in zip(range(len(ind0)), digits.images[ind0], digits.target[ind0]):
        ax[ind1].axis('off')
        ax[ind1].imshow(image)
        ax[ind1].set_title('label: {}'.format(label))


if __name__ == "__main__":
    show_digits()
