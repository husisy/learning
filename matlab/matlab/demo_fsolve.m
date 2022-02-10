hf1 = @(x1,x2) exp(-exp(-(x1+x2))) - x2.*(1+x1.^2);
hf2 = @(x1,x2) x1.*cos(x2) + x2.*sin(x1) - 0.5;
hf3 = @(x) [hf1(x(1),x(2)), hf2(x(1),x(2))];

%%
hFig = figure();

N1 = 100;
N2 = 100;
x = linspace(0,1,N1);
y = linspace(0,1,N2);
fval1 = hf1(x.',y).^2;
fval2 = hf2(x.',y).^2;

hAxes(1) = subplot(1,2,1);
surf(x,y,fval1);
shading interp;

hAxes(2) = subplot(1,2,2);
surf(x,y,fval2);
shading interp;

%%
options = optimoptions('fsolve','Display','none','PlotFcn',@optimplotfirstorderopt);
[x0,fval0] = fsolve(hf3, [0,0], options);
