import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import qiskit
import qiskit_machine_learning.datasets


def show_adhoc_dataset():
    train_data, train_labels, test_data, test_labels, sample_total = qiskit_machine_learning.datasets.ad_hoc_data(
                    training_size=20, test_size=5, n=2, gap=0.3, include_sample_total=True, one_hot=False)

    fig,(ax0,ax1,ax2) = plt.subplots(1, 3, figsize=(15,5))
    ax0.set_title("Data")
    ax0.set_ylim(0, 2 * np.pi)
    ax0.set_xlim(0, 2 * np.pi)
    ind0 = train_labels == 0
    ind1 = np.logical_not(ind0)
    ind2 = test_labels == 0
    ind3 = np.logical_not(ind2)
    ax0.scatter(train_data[ind0,0], train_data[ind0,1], marker='s', facecolors='w', edgecolors='C0', label="A train")
    ax0.scatter(train_data[ind1,0], train_data[ind1,1], marker='o', facecolors='w', edgecolors='C3', label="B train")
    ax0.scatter(test_data[ind2,0], test_data[ind2,1], marker='s', facecolors='C0', label="A test")
    ax0.scatter(test_data[ind3,0], test_data[ind3,1], marker='o', facecolors='C3', label="B test")
    ax0.legend()

    cmap = matplotlib.colors.ListedColormap(["C3","w","C0"])
    ax1.set_title("Class Boundaries")
    ax1.set_ylim(0, 2 * np.pi)
    ax1.set_xlim(0, 2 * np.pi)
    ax1.imshow(sample_total.T, interpolation='nearest', origin='lower', cmap=cmap, extent=[0, 2*np.pi, 0, 2*np.pi])

    ax2.set_title("Data overlaid on Class Boundaries")
    ax2.set_ylim(0, 2 * np.pi)
    ax2.set_xlim(0, 2 * np.pi)
    ax2.imshow(sample_total.T, interpolation='nearest', origin='lower', cmap=cmap, extent=[0, 2*np.pi, 0, 2*np.pi])
    ax2.scatter(train_data[ind0,0], train_data[ind0,1], marker='s', facecolors='w', edgecolors='C0', label="A")
    ax2.scatter(train_data[ind1,0], train_data[ind1,1], marker='o', facecolors='w', edgecolors='C3', label="B")
    ax2.scatter(test_data[ind2,0], test_data[ind2,1], marker='s', facecolors='C0', edgecolors='w', label="A test")
    ax2.scatter(test_data[ind3,0], test_data[ind3,1], marker='o', facecolors='C3', edgecolors='w', label="B test")
