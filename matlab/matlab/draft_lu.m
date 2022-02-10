hfe = @(x,y) max(reshape(abs(x-y)./(abs(x)+abs(y)+1e-5), [], 1));

N0 = 23;
mat0 = sprand(N0, N0, 0.9);
[matL,matU,matP,matQ] = lu(mat0);
assert(hfe(matL, tril(matL)) < 1e-7);
assert(hfe(matU, triu(matU)) < 1e-7);
assert(hfe(matP*matP', eye(N0))<1e-7);
assert(hfe(matP'*matL*matU*matQ', mat0) < 1e-7);
[indP,~,~] = find(matP);
[indQ_reverse,~,~] = find(matQ');
assert(hfe(matL(indP,:)*matU(:,indQ_reverse), mat0)<1e-7);
