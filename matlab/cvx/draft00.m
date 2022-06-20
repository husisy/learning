m = 20; n = 10; p = 4;
A = randn(m,n); b = randn(m,1);
C = randn(p,n); d = randn(p,1); e = rand();
cvx_begin
    variable x(n)
    minimize( norm( A * x - b, 2 ) )
    subject to
        C * x == d
        norm( x, Inf ) <= e
cvx_end


%% least squares
m = 16; n = 8;
A = randn(m,n);
b = randn(m,1);
cvx_begin
    variable x(n)
    minimize( norm(A*x-b) )
cvx_end
% x_ls = A \ b; %pseudo-inverse


%% bound-constrained least squares
m = 20; n = 10;
A = randn(m,n); b = randn(m,1);
bnds = randn(n,2);
l = min( bnds, [], 2 );
u = max( bnds, [], 2 );
cvx_begin
    variable x(n)
    minimize( norm(A*x-b) )
    subject to
        l <= x <= u
cvx_end
%x_qp = quadprog( 2*A'*A, -2*A'*b, [], [], [], [], l, u ); %matlab optimization toolbox


%% infinity-norm function
m = 20; n = 10;
A = randn(m,n); b = randn(m,1);
f    = [ zeros(n,1); 1          ];
Ane  = [ +A,         -ones(m,1)  ; ...
         -A,         -ones(m,1) ];
bne  = [ +b;         -b         ];
xt = linprog(f,Ane,bne); %matlab optimization toolbox
x_cheb = xt(1:n,:);
cvx_begin
    variable x(n)
    minimize( norm(A*x-b,Inf) )
cvx_end


%% 1-norm
m = 20; n = 10;
A = randn(m,n); b = randn(m,1);
f    = [ zeros(n,1); ones(m,1);  ones(m,1)  ];
%Aeq  = [ A,          -eye(m),    +eye(m)    ];
%lb   = [ -Inf(n,1);  zeros(m,1); zeros(m,1) ];
%xzz  = linprog(f,[],[],Aeq,b,lb,[]); %matlab optimization toolbox
%x_l1 = xzz(1:n,:) - xzz(n+1:end,:);
cvx_begin
    variable x(n)
    minimize( norm(A*x-b,1) )
cvx_end


%% strange-norm
m = 20; n = 10;
A = randn(m,n); b = randn(m,1);
k = 5;
cvx_begin
    variable x(n);
    minimize( norm_largest(A*x-b,k) );
cvx_end
% norm_largest(A*x-b,k) %cvx_optval
