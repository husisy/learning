using LinearAlgebra

z0 = [1 2 3; 4 1 6; 7 8 1]
# z0 = Int64[1 2 3; 4 1 6; 7 8 1]
# NOT equal to [[1,2,3], [4,1,6], [7,8,1]]
tr(z0)
det(z0)
inv(z0)


z0 = Float64[-4 -17; 2 2]
eigvals(z0)
eigvecs(z0)


z0 = [1.5 2 -4; 3 -1 -6; -10 2.3 4] #not hermitian, symmetric, triangular, tridiagonal, bidiagonal
# bad bahevior if Symmetric(non-symmetric-matrix)
L,U = factorize(z0)
z1 = Symmetric([1.5 2 -4; 2 -1 -3; -4 -3 5])
D,U,P = factorize(z1)
