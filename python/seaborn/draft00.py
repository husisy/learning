import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.ion()

sns.set_style('darkgrid')
penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")

np0 = penguins['bill_length_mm'].values
np1 = penguins['bill_depth_mm'].values
sns.jointplot(x=np0, y=np1)
fig = plt.gcf()
ax = fig.get_axes()[0]
ax.set_xlabel('bill_length_mm')
ax.set_ylabel('bill_depth_mm')
fig.tight_layout()



tips = sns.load_dataset('tips')
sns.scatterplot(data=tips, x="total_bill", y="tip")

tmp0 = np.meshgrid(np.arange(5), np.arange(5), indexing='ij')
np0 = tmp0[0].reshape(-1)
np1 = tmp0[1].reshape(-1)
np2 = np.random.randint(1, 10, size=np0.shape)
sns.scatterplot(x=np0, y=np1, size=np2)
