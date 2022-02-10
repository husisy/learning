# kwant

1. link
   * [official site](https://kwant-project.org/)
   * [documentation](https://kwant-project.org/doc/1/)
   * [tutorial](https://mybinder.org/v2/gh/kwant-project/kwant-tutorial-2016/master)
   * [gitlab](https://gitlab.kwant-project.org/kwant/kwant)
   * kwant a software for quantum transport [NJP2014](https://iopscience.iop.org/article/10.1088/1367-2630/16/6/063065)
2. install `conda install -c conda-forge kwant`
3. $t=\frac{\hbar^2}{2ma^2}$
4. 有趣的vector potential [link](https://kwant-project.org/doc/1/tutorial/spin_potential_shape#nontrivial-shapes)
   * vector potential $A_x(x,y)=\Phi\delta(x)\Theta(-y)$
   * magnetic field $B_z(x,y)=\Phi\delta(x)\delta(y)$
5. normal metal, superconductor
   * N-S interface conductance $G=\frac{e^2}{h}(N-R_{ee}+R_{he})$
   * the number of electron channel $N$
   * the total probability of reflection from electrons to electrons in the normal lead $R_{ee}$
   * the total probability of reflection from electrons to holes in the normal lead $R_{he}$
   * normal metal: electron-hole conservation law, particle-hole symmetry
   * conductance that is proportional to the square of the tunneling probability within the gap, and proportional to the tunneling probability above the gap. At the gap edge, we observe a resonant Andreev reflection
6. discrete symmetry: time-reversal, particle-hole and chiral
7. local density $\rho_a=\psi^\dagger M_a \psi_i$
8. local current $J_{ab}=i(\psi_b^\dagger H_{ab}^\dagger M\psi_a - \psi_a^\dagger MH_{ab}\psi_b)$
9. the discretization of a continuum model is an approximation that is only valid in the low-energy limit
10. site: `.family`, `.tag`, `.pos`
11. lattice: `Monatomic`, `Polyatomic`

```bash
conda create -y -n kwant
conda install -y -n kwant -c conda-forge cudatoolkit=11.3
conda install -y -n kwant -c pytorch pytorch torchvision torchaudio
conda install -y -n kwant -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cupy kwant holoviews
```

two-dimensional electron gas

$$H = -\frac{\hbar^2}{2m}(\partial_x^2+\partial_y^2) + V(y)$$

Rashba SOI, Zeeman splitting, [PRL2003](http://prl.aps.org/abstract/PRL/v90/i25/e256601) [nature2010](http://www.nature.com/nphys/journal/v6/n5/abs/nphys1626.html)

$$H = -\frac{\hbar^2}{2m}(\partial_x^2+\partial_y^2) - i\alpha(\partial_x\sigma_y-\partial_y\sigma_x) + V(y)$$

Bogoliubov-de Gennes (BdG) Hamiltonian, time reversal operator $\mathcal{T}$

$$
H=\begin{pmatrix}
   H_0-\mu & \Delta \\
   \Delta^\dagger & \mu-\mathcal{T}H_0\mathcal{T}^{-1}
\end{pmatrix}
$$

spinful Hamiltonian

$$
H=-t\sum_{\langle ij \rangle,\alpha} {|i\alpha\rangle\langle j\alpha|} + J\sum_{i,\alpha,\beta}{\vec{m}_i\cdot \vec{\sigma}_{\alpha\beta} |i\alpha\rangle\langle i\beta|}
$$

magnetic texture

$$
\vec{m}_i=\left( \frac{x_i}{x_i^2+y_i^2}\sin\theta_i, \frac{y_i}{x_i^2+y_i^2}\sin\theta_i, \cos\theta_i \right)
$$

$$
\theta_i=\frac{\pi}{2} (\tanh((r_i-r_0)/\delta)-1)
$$

## kernel polynomial method

1. spectral density $\rho_A(E)=\rho(E)A(E)$
   * energy $E$
   * Hilbert space operator $A$
   * expectation value of $A$ for all the eigenstates of the Hamiltonian $H$ with energy $E$
   * density of state $\rho(E)=Tr(\delta(E-H))=\sum_k{\delta(E-E_k)}$
2. pros and cons
   * pros: suited for large systems: not interested in individual eigenvalues, but rather in obtaining an approximate spectral density
   * accuracy: controlled by the number of the moments, the lowest accuracy is at the center of the spectrum, more accurate at the edges of the spectrum
   * Jackson kernel $\sigma=\pi a/N$
   * random vectors will explore the range of the spectrum. The bigger the system is, the number of random vectors required reduces
   * NO noise for local vectors
3. optimal set of energy is not evenly distributed
4. boundary of the spectrum using `scipy.sparse.linalg.eigsh`
5. sesquilinear map
6. Kubo conductivity must be normalized with the area covered by the vectos
