clear all

load results_AE5.mat

gbest = gbest';
save input_latent.dat -ascii gbest
system('python vae_generator.py')
coord = load('shape.dat');

s_mean = [0,0,0,0]';
save input_latent.dat -ascii s_mean
system('python vae_generator.py')
coord_mean = load('shape.dat');

% 0012
s0012 = [0.170374, 0.160207, 0.143643, 0.166426, 0.110476, 0.179433, ...
    -0.170374, -0.160207, -0.143643, -0.166426, -0.110476, -0.179433];

figure;
% optimized shape
plot(coord(:,1),coord(:,2));
hold on
plot(coord_mean(:,1),coord_mean(:,2),'-');
hold on
coord = CST_airfoil(s0012(1:length(s0012)/2),s0012(length(s0012)/2+1:length(s0012)),0,200);
plot(coord(:,1),coord(:,2),'-');
hold on
legend('optimized geometry','mean geometry','0012 geometry');
axis([0 1 -0.12 0.12])

% compute value
AOA = 3;
[Integral_Spp, OASPL, Cl, Cd] = change_shape(gbest,AOA,0);
OASPL
Cl
Cd

[Integral_Spp, OASPL, Cl, Cd] = change_shape(s_mean,AOA,0);
OASPL
Cl
Cd

[Integral_Spp, OASPL, Cl, Cd] = change_shape_CST(s0012,AOA,0);
OASPL
Cl
Cd

