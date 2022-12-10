import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import time

def demo_jupyter_dynamically_plot():
    # run it in jupyter code block
    # https://stackoverflow.com/questions/39658717/plot-dynamically-changing-graph-using-matplotlib-in-jupyter-notebook
    fig,ax = plt.subplots()
    hdisplay = IPython.display.DisplayHandle()
    hdisplay.display(fig)
    np_rng = np.random.default_rng()
    for i in range(10):
        ax.clear()
        ax.plot(np_rng.uniform(0, 1, size=5))
        hdisplay.update(fig)
        # IPython.display.update_display(fig, display_id=hdisplay.display_id)
        time.sleep(0.1)
    plt.close(fig)
