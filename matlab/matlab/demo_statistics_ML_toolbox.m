% negative binomial log-likehood
syms x y
expr1 = log(x/(1-x));
expr2 = log(1+exp(-y*expr1));
simplify(subs(expr2, y, sym(1)))
simplify(subs(expr2, y, sym(-1)))


% beta distribution
x = linspace(0,1,100);
y1 = betapdf(x,0.75,0.75);
y2 = betapdf(x,1,1);
y3 = betapdf(x,4,4);

hFig = figure();
axes(hFig);
color1 = zcColor(1);

hLine(1) = line(x,y1,'Color',color1(1,:),'LineWidth',2);
hLine(2) = line(x,y2,'Color',color1(2,:),'LineStyle',':','LineWidth',2);
hLine(3) = line(x,y3,'Color',color1(3,:),'LineStyle','-.','LineWidth',2);
tmp1 = {'(0.75,0.75)','(1,1)','(4,4)'};
hLegend = legend(hLine, tmp1,'Location','NorthEast','Color','none','Box','off');

num1 = 1e4;
x1 = betarnd(3, 5, num1, 1);
phat = betafit(x1);
x2 = linspace(0,1,100);
y2 = betapdf(x2, phat(1),phat(2));

hFig = figure();
hAxes = axes(hFig);
color1 = zcColor(1);

hHist = histogram(x1,40,'Normalization','pdf');
hHist.FaceColor = color1(2,:);
hHist.EdgeColor = 'none';
hLine = line(x2,y2,'Color',color1(1,:),'LineWidth',2);
% hBar = bar(hAxes,xCenter,y1/max(y1));
% hBar.FaceColor = color1(2,:);
% hBar.EdgeColor = 'none';
% hLine = line(xCenter,y2/max(y2),'Color',color1(1,:),'LineWidth',2);
