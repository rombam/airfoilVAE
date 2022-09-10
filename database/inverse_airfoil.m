% PSO optimization of geometry
% modify change_shape.m to use autoencoder

clear all

% addpath('supporters_Amiet\')

try 
    % delete file
    delete POLAR_OUTPUT
    delete doesntmatter
catch
end

nx = 6; % nx CST parameters
AOA = 3;
% should think of ways to limit geometry
% change BSJ for different objectives

% G: number of iterations
% nx: swarm length (number of geometry)
% m: swarm size
% w,c1,c2 e PSO parameters: w is inertial weight, c1 c2 learning factors
G = 240;
m = 50;
w = 0.6*ones(G,1);
c1 = 2;
c2 = 2;
wmin=0.40;
wmax=0.9;

s0012 = [0.170374, 0.160207, 0.143643, 0.166426, 0.110476, 0.179433, ...
    -0.170374, -0.160207, -0.143643, -0.166426, -0.110476, -0.179433];
s0008 = 0.8/1.2*s0012;
[Integral_Spp, OASPL0, Cl0, Cd0, coord0] = change_shape_CST(s0012, AOA, 0);


% range of s 
% MinX = [1e-2,1e-2,1e-2,1e-2];
% MaxX = [0.5,0.5,0.5,0.5];

% naca0012 +- 30%
% MinX = [s0012(1:nx/2) * 0.7, -s0012(1:nx/2) * 1.3];
% MaxX = [s0012(1:nx/2) * 1.3, -s0012(1:nx/2) * 0.7];
% Xbounds = (MaxX-MinX);

% MinX = [0,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.7,-0.7,-0.5,-0.5,-0.5];
% MaxX = [0.5,0.7,0.7,0.5,0.5,0.5,0,0.5,0.5,0.5,0.5,0.5];

% give CST parameter range
% MinX = [s0012(1:nx/2),s0012(7:6+nx/2)]-0.2;
% MaxX = [s0012(1:nx/2),s0012(7:6+nx/2)]+0.2;
% MinX(1) = 0;
% MaxX(4) = 0;

MinX = [-3,-3,-3,-3,-3,-3];
MaxX = [3,3,3,3,3,3];
MinV = -0.4*(MaxX-MinX);
MaxV = 0.4*(MaxX-MinX);
Xbounds = (MaxX-MinX);
MeanX = (MinX + MaxX) / 2;

shaperef = load('shape_optim.dat');

Cl0
% can define initial geometry
for i = 1:m
    pop(i,:) = MeanX + rands(1,nx).* Xbounds/2;
end

for i = 1:m % calibrate random number
    for j = 1:nx
        if pop(i,j) < MinX(j)
            pop(i,j) = MinX(j);
        end
        if pop(i,j) > MaxX(j)
            pop(i,j) = MaxX(j);
        end
    end
end

for i = 1:m
    V(i,:) = rands(1,nx).* 0.2 .* Xbounds;
end

% evaluate fitness / objective function to find the individual best and the group best
for s = 1:m
    indivi = pop(s,:); % random particle
%     coord = CST_airfoil(indivi,-indivi,0,80);
    % fitness function
    indivi
    s_latent = indivi';
    save input_latent.dat -ascii s_latent;
    system('python vae_generator.py');
    shapepso = load('shape.dat');
    BSJ = mse(shaperef(:,2)-shapepso(:,2));
    Error(s) = BSJ; % saving error / fitness
end

[OderEr,IndexEr] = sort(Error); % ascend noise
Error;
Errorleast = OderEr(1); % smallest noise ERRORleast
for i = 1:m
    if Errorleast == Error(i)
        gbest = pop(i,:); % best in swarm
        break;
    end
end
ibest = pop; % individual best


for kg = 1:G % kg is current generation
    kg
    
    for s = 1:m
    % 4% chance of mutation       
        for j = 1:nx
            for i = 1:m
                if rand(1)<0.02
                    pop(i,j) = MinX(j) + abs(rands(1)).* (MaxX(j)-MinX(j)); %rand(1);
                end
            end
        end
    % r1,r2 are PSO parameters        
        r1 = rand(1);
        r2 = rand(1);

    % update velocity and individual 
        w(kg)=wmax-((wmax-wmin)/G)*kg;  % reduce weight as optimization progresses
        V(s,:) = w(kg)*V(s,:) + c1*r1*(ibest(s,:)-pop(s,:)) + c2*r2*(gbest-pop(s,:));
        
        % limit velocity
        for j = 1:nx
            if V(s,j) < MinV(j)
                V(s,j) = MinV(j);
            end
            if V(s,j) > MaxV(j)
                V(s,j) = MaxV(j);
            end
        end
        
        pop(s,:) = pop(s,:) + V(s,:);
        
        for j = 1:nx
            if pop(s,j) < MinX(j)
                pop(s,j) = MinX(j);
            end
            if pop(s,j) > MaxX(j)
                pop(s,j) = MaxX(j);
            end
        end

    % objective function after the update
%     pop(s,:)
    
    s_latent = pop(s,:)';
    save input_latent.dat -ascii s_latent;
    system('python vae_generator.py');
    shapepso = load('shape.dat');
    BSJ = mse(shaperef(:,2)-shapepso(:,2));

    
        err(s) = BSJ;
        
    % update individual best and group best, compare current fitness with historical best     
        if err(s)<Error(s)
            ibest(s,:) = pop(s,:);
            Error(s) = err(s);
        end
        if err(s)<Errorleast % compare with swarm best
            gbest = pop(s,:);
            Errorleast = err(s);
        end
%         if Errorleast<0.002
%             break;
%         end
    end
    ibest
    gbest
    Best(kg) = Errorleast
end
    shape = gbest       % global optimal shape

save results.mat
shapet = shape';
save input_latent_optim.dat -ascii shapet;

%    2.0452000e+00
%    6.6550000e-01
%    1.2801000e+00
%   -1.8235000e+00

%   2.1661000e+00   7.1990000e-01   1.5835000e+00  -1.8391000e+00

% 6:-1.7045   -0.5734   -0.2972   -1.5920    0.3666   -0.2395