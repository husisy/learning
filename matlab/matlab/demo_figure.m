% figure0
t = 0:pi/10:2*pi;
[X,Y,Z] = cylinder(4*cos(t));

hFig = figure();
hAxes = gobjects(2,2);

hAxes(1,1) = subplot(2,2,1); mesh(X); title('X');
hAxes(1,2) = subplot(2,2,2); mesh(Y); title('Y');
hAxes(2,1) = subplot(2,2,3); mesh(Z); title('Z');
hAxes(2,2) = subplot(2,2,4); mesh(X,Y,Z); title('X,Y,Z');


% figure1
fig = figure('Position', [1,1,600,600], 'Color', 'none');
ax = axes(fig);

num1 = 5000;
r = rand([num1,1]);
r = (1 - sqrt(1-r.^2) + r)/2;
theta = rand([num1,1])*2*pi;
x = r.*cos(theta);
z = r.*sin(theta);
% x = rand([num1,1])*2 - 1;
% z = rand([num1,1])*2 - 1;
% ind1 = (x.^2 + z.^2)<1;
% x = x(ind1);
% z = z(ind1);
y = sqrt(1 - x.^2 - z.^2);
s = my_rand(size(x,1));
hs = scatter3(x,y,z,s,'filled','MarkerFaceColor',[1,1,1]);

axis equal
ax.XLim = [-1.1,1.1];
ax.YLim = [-1.1,1.1];
ax.ZLim = [-1.1,1.1];
ax.XAxis.Visible = 'off';
ax.YAxis.Visible = 'off';
ax.ZAxis.Visible = 'off';
ax.Color = 'none';
view(90, 60);


function ret = my_rand(num1)
ret = abs(randn([num1*5,1]))*3;
ret = ret(ret>1 & ret<30);
ret = ret(1:num1);
end
