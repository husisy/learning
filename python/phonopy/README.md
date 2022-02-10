# phonopy

见[github-husisy/pyphonopy](https://github.com/husisy/pyphonopy)

## ws00

1. `phonopy --fc vasprun.xml`
   * 输入：`vasprun.xml`
   * 输出：`FORCE_CONSTANTS`
2. `phonopy -c POSCAR-unitcell band.conf`
   * 输入：`POSCAR-unitcell`, `band.conf`
   * 输出：`band.yaml`, `phonopy.yaml`，后者好像没啥用
3. `phonopy-bandplot --gnuplot > band.dat`：可选，生成绘图数据
   * 输入：`phonopy.yaml`
   * 输出：`band.dat`

## integer quadratic optimization

给定实数域上对称正定矩阵$A\in R^{3\times 3}$，给定实数域上矢量$\vec{r}\in \mathcal{R}^{3\times 1}$，求解整数域上“矢量”$\vec{n}\in \mathcal{Z}^{3\times 1}$，使得

$$
\min_{\vec{n}} ( \vec{n}-\vec{r} ) ^TA ( \vec{n}-\vec{r} )
$$

解可能不唯一，找到其中之一即可。举个栗子：

$$
A=\begin{bmatrix}
1&11&\\
11&122& \\
&&1\\
\end{bmatrix}
$$

$$\vec{r}=\begin{pmatrix}0.4&0.6&0\end{pmatrix} ^T$$

$$\vec{n}=\begin{pmatrix}-4&1&0\end{pmatrix}^T$$

**原物理问题**：欧几空间（定义了距离），选取一组非正交非归一基矢$P=[\vec{e}_1,\vec{e}_2,\vec{e}_3]$，定义**格点**为$n_1\vec{e}_1+n_2\vec{e}_2+n_3\vec{e}_3$，其中$n_1,n_2,n_3$为整数。给定空间中的一点$\vec{r}=r_1\vec{e}_1+r_2\vec{e}_2+r_3\vec{e}_3$，求解与该点最近的格点。易证此处的$P$与上文中的$A$满足关系

$$A=P^TP$$
