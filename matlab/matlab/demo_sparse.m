n1 = 10;
% z1 = sparse(2:n,1:n-1,ones(1,n-1),n,n);
hfe = @(x,y) max(max(abs(x-y)./(abs(x)+abs(y)+1e-3)));

load west0479
z1 = west0479;
spy(z1);
[ind1,ind2,v] = find(z1);
[m,n] = size(z1);
z2 = sparse(ind1,ind2,v,m,n);
disp(['hfe: ',num2str(hfe(z1,z2))])
