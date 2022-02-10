%% random
rand(1,3); %the same sequence every time restarting matlab
rng(); %get the current random number generator, seed should be 0 after restarting
rng(233); %change seed to 233
rng('shuffle'); %change to random seed, relate to time
rng('default'); %change seed to 0

% not true random number, no
z0 = zeros(5,1);
for ind0 = 1:size(z0,1)
    pause(1);
    rng('shuffle');
    tmp0 = rng();
    z0(ind0) = tmp0.Seed();
end
disp(z0(2:end) - z0(1:(end-1))) %close to 100

% maybe a good enough rng, no corresponding as os.urandom() in python
rng(uint32(bitshift(bitshift(tic(), 32), -32)));

% for function design (without effecting)
previousSettings = rng(233);
rng(previousSettings)


%% profile
