%% run(test_test())
function ret = test_linalg()
    ret = functiontests(localfunctions);
end

function test_chol(testCase)
    hfe = @(x,y) max(reshape(abs(x-y)./(abs(x)+abs(y)+1e-5), [], 1));
    N0 = 10;
    matA = real(generate_hermite_matrix(N0, 1, 2));
    matL = chol(matA);
    assert(hfe(matA, matL'*matL) < 1e-7);
end

function ret = generate_hermite_matrix(N, min_eig, max_eig)
if nargin==1
    min_eig = 1;
    max_eig = 2;
end
if nargin==2
    max_eig = min_eig + 1;
end
tmp0 = rand(N, N);
EVC = expm(1j*(tmp0 + tmp0')/2);
EVL = rand(N,1) * (max_eig-min_eig) + min_eig;
ret = EVC' * (EVL.*EVC);
end
