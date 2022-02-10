# GTPack

1. link
   * [official-site](https://gtpack.org/)
   * [tutorial](https://gtpack.org/tutorials/)
   * [documentation](https://gtpack.org/documentation/)
2. installation
   * pass

```MMA
Needs["GroupTheory`"]
GTWhichRepresentation[]
GTGetMatrix[C3z] // MatrixForm

GTChangeRepresentation["SU(2)"]
C2x \[SmallCircle] C2x

GTChangeRepresentation["SU(2)xS"]
GTGetMatrix[C3z]
GTGetMatrix[IC3z]
```
