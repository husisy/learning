hf_hermite = @(x) (x+x')/2;

gpuDevice
% gpuDevice(2); %use specific device

x_cpu0 = rand(3,3);
x_gpu0 = gpuArray(x_cpu0);
x_cpu1 = gather(x_cpu0);
whos x_cpu0 x_cpu1 x_gpu0

x0 = ones([3,3], "gpuArray");
x1 = gpuArray.linspace(0, 1, 10);
x2 = complex(gpuArray(rand(3,3)));
x3 = eye(3, "int32", "gpuArray");
x4 = zeros([3,3], "gpuArray");
x5 = rand([3,3], "gpuArray");
% underlyingType
size(x0);

x = gpuArray(hf_hermite(rand(3,3)));
expm(x)
diag(x)


mat0 = gpuArray(rand(5, 5));
ind0 = gpuArray(rand(1,5)<0.5); %boolean indexing
mat1 = mat0(:, ind0);
disp(class(mat0)) %gpuArray
disp(class(mat1)) %gpuArray whether ind0 is gpuArray or not
