# VASP

1. link
   * [sobereva-VASP安装方法](http://sobereva.com/455)
   * [学术之友-给真小白看的VASP本地编译自学指南](https://mp.weixin.qq.com/s?__biz=MzI2OTQ4OTExOA==&mid=2247484430&idx=1&sn=3dfcb85faac6cd6b8c3b4e4dcd303149&chksm=eadec2bfdda94ba9b4d45fa0ecc744879bb1013fd6bb83ac9da49a32c1a76c873ed8dbff5def&mpshare=1&scene=1&srcid=0106RxMPU2pbSwmhSKYSYxGC&sharer_sharetime=1579398113698&sharer_shareid=6fc201bb8b325af9f67740b741ef6a09&key=56a81ade6ca7867daa1dca9a6168eec3c855469dd2f1e9817d74cc5082a9688adf10c42de6637f9125ebdd36fae00e8a4060e86e4824ada5b9972fa640a70b2557e454686a2dbbb0c8a9d5008800e0c8&ascene=1&uin=MjI5NzA3OTgzOQ%3D%3D&devicetype=Windows+10&version=6208006f&lang=zh_CN&exportkey=AbYI%2FMSkFESwbIQlzsl3TTw%3D&pass_ticket=jJb%2Bs1s3SmwJ%2Fxd0JKpJ1hUtvuX%2BIBNVqTtsBuGyAZpv1Fz5J93cZVtcQfRiADyj)
2. 以下所有参数可能会对于不同版本VASP有不同的设置，这里只使用来源于Vienna, April 20的最新版本VASP
3. note from Wenlong E, 3_Au111
4. PREC(Normal): Low | Medium | High(不推荐) | Normal(推荐) | Accurate(精确计算) | Single , 影响ENCUT(未给出时), NGX, NGY, NGZ, NGXF, NGYF, NGZF ROPT； 推荐使用Normal; Accurate(涉及精确的弹性性质的计算时使用)可以消除wrap arounding error(进一步提高力精度ADDGRID = .TRUE.)。
5. ISTART: 0 | 1 | 2;0, 从0开始； 1, 存在WAVECAR, 可以修改size/shape/ENCUT, 会相应的生成平面波基组, 是否可以修改k-point mesh但数量不变以及其他一些什么(存疑，见manual Page60)； 2， 一般不需要使用，在计算平衡体积保持平面波基组不变的时候使用； 3，MD使用
6. ICHARG: 0 | 1 | 2 | 4； 0， 存在有效的WAVECAR时，从initial orbitals构建电荷密度； 1， 从CHGCAR插值构建电荷密度，可能存在问题increase LMAXMIX to 4(d-elements) or even 6 (f-elements)； 2， 从原子电荷密度构建体系电荷密度； 4，从POT.文件中读取potential，与LVTOT = .TRUE.有关； BELOW CHARGE DENSITY WILL BE KEPT CONSTANT DURING THE WHOLE ELECTRONIC MINIMIZATION; 11， 从CHGCAR获得特征值（能带）或态密度; 12; 计算力和stress，与MD有关
7. ENCUT=400; accurate精度设置为比POTCAR中的最大ENCUT大30%，一般文献中会给出
8. INIWAV: 0 | 1(推荐); 0, 果胶模型，但数据库不完全兼容，也没有必要； 1， 随机初始波函数
9. NELM: 整数(60)； 电子自洽最大步数，一般40步没有收敛，很有可能就达不到收敛，考虑修改 IALGO, (ALGO), LDIAG ,或者使用 the mixing-parameters
10. NELMIN: 整数(2)(弛豫推荐6)； 电子自洽最少步数，一般不需修改； 计算MD或者离子弛豫时建议设置为4-8
11. NELMDL: 非正整数(-5);对于随机波函数，延迟更新电荷密度的步数
12. EDIFF: 1e-4; 电子自洽能量该变量小于该值时迭代结束，无需更小。
13. LREAL： .FALSE.(default) | .TRUE. | Auto（推荐）; 投影算符在实空间还是倒易空间投影
14. ALGO = Normal(default) | VeryFast | Fast | Conjugate | All | Damped | Subrot | Eigenval | None | Nothing | Exact | Diag; Normal, blocked Davidson iteration scheme; VeryFast, RMM-DIIS; Fast, 使用Normal和VeryFast两者结合的算法； Exact/Diag, if more than 30-50 % of the states are calculated (e.g. for GW or RPA calculations); Eigenval, recalculate; 这里具体这么选择看manual Page83
15. EDIFFG = EDIFF*10; 正表示离子弛豫迭代所达到的最小变化能量，负表示最小受力。
16. NSW， 0（default); 离子最大迭代步数。
17. IBRION= -1（NEW=0,default） | 0(default) | 1 | 2(推荐) | 3 | 5 | 6 | 7 | 8 | 44； -1， 离子不动； 0, MD; 2, conjugate gradient algorithm, 需要设置POTIM； 3, Damped molecular dynamics， 当初始值很糟糕时使用， POTIM， SMASS； 1, RMM-DIIS(quasi-Newton)，在最小值附近时使用， 谨慎设置POTIM, NFREE； 5/6， IBRION=5 and IBRION=6 are using finite differences to determine the second derivatives (Hessian matrix and phonon frequencies)； 7/8， density functional perturbation theory to calculate the derivatives
18. POTIM， 0.5(default); 对于共轭梯度算法，0.5可以接受，但是newton 和 damped方法可能会偏大而不稳定。
19. ISIF： Before you perform relaxations in which the volume or the cell shape is allowed to change you must read and understand section 7.6. In general volume changes should be done only with a slightly increased energy cutoff (i.e. ENCUT=1.3 * default value , or PREC=High in VASP.4.4)
20. ISYM= -1 | 0 | 1(USPP default) | 2(PAW default) | 3; 3, force and stress tensor对称，charge density不对称（当弛豫需要保持晶体对称性时使用）； 对称性有POSCAR和MAGMOM决定
21. EMIN: -(lowest KS-eigenvalue - ∆)
22. EMAx: (highest KS-eigenvalue + ∆)，比EMIN小时，计算所有
23. NEDOs: 301
24. ISMEAR = -5 | -4 | -3 | -2 | 0 | N； 默认1； −1 Fermi-smearing； 1..N method of Methfessel-Paxton order N（占据数可能为负）； -2，占据数在INCAER或WAVECAR中指定（FERWE,FERDO）; −3 perform a loop over smearing-parameters supplied in the INCAR file; −4 tetrahedron method without Blochl corrections; −5 tetrahedron method with Blochl corrections;
    * 对于bulk计算总能和态密度，推荐使用-5，但对于金属计算force and stress不准确；对于半导体和绝缘体，力的计算是准确的，但占据数存在问题；对于基于force计算的身子谱，推荐使用ISMEAR>0; MP方法需要谨慎选取SIGMA，SIGMA的选择保证OUTCAR文件中的free energy和total energy(entropy T*S)的差足够小,对于半导体和绝缘体禁止使用MP方法，只能使用-5和0
    * 半导体，绝缘体使用-5，0（SIGMA=0.05）(对于-5，k点必须能够连成四面体，所以太少k点或者k只在一个平面，计算能带的k点都在一条线上)
    * 金属弛豫使用1，2配合合适的SIGMA(默认0.2还算合理)(the entropy term should be less than 1 meV per atom)
    * 计算态密度或者总能都可以使用-5
25. LWAVE = .TRUE.(default) | .FALSE.
26. LCHARG = .TRUE. | .FALSE.

```bash
ISPIN=1
LNONCOLLINEAR=F # non collinear calculations
LSORBIT=F #spin-orbit coupling
INIWAV=1 #electr: 0-lowe 1-rand 2-diag
LASPH=F #aspherical Exc in radial PAW
LMAXPAW=-100 #max onsite density
VOSKOWN= 0 #Vosko Wilk Nusair interpolation
LVDW=TRUE
LVTOT=F
LELF=F
LORBIT=0
NWRITE=1
```
