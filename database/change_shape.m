%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main script to calculate the trailing edge noise at different
% directivity angles based on the theory of Roger-Moreu 2005. 
% In this code, the back scattering effect is also taken into account. 
% Laura Botero Bolívar - University of Twente 
% l.boterobolivar@utwente.nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Integral_Spp, OASPL, Cl, Cd, coord, constraint, valid_res] = change_shape(s,AOA,mode)
% possible outputs: [Integral_Spp, OASPL, Cl, Cd]

% valid_res: whether the result is valid (valid_res = 0, not convergence)

%% generate geometry
s_latent = s';
save input_latent.dat -ascii s_latent;
system('python vae_generator.py');
coord = load('shape.dat');

% add thickness constraint
constraint = 1;
% s = min([0.06-max(abs(coord(:,2))),0]);
n = size(coord,1)-1;
s = min([0.12-max(coord(2:n/2,2)-coord(n/2+2:end-1,2)),0]);
% check if lower surface becomes upper
s2 = min(coord(2:n/2,2)-coord(end-1:-1:n/2+2,2));

if mode == 1 && (s < 0 || s2 < 0) % check constraint (thickness)
   constraint = 0;
   Integral_Spp = 0;
   OASPL = 0;
   Cl = 0;
   Cd = 0;
   valid_res = 0;
else

%% Set the case
% clear all
[fluid,input] = inputs_definition();
R = input.z;
%% specifiy frequency
kc = 1;
k = kc/input.chord;
omega_d = k*fluid.c0;

A = input.A';

for i = 1
input.AoA = AOA;
%% Trailing edge noise prediction
[omega, S_pp(:,i), Phi_pp(:,i),Cl(i),Cd(i), Cdp(i), Cm(i), Cf(:,i), valid_res] = TE_noise_Prediction(input,fluid);

if valid_res == 0
    Integral_Spp=0; OASPL = 0;
else
Integral_Spp=abs(trapz(omega/(2*pi),10*log10(S_pp(:,i)/(20*10^-5))));

%% OASPL: calculation
% % old
% Spp_db(:,i) = 10*log10(Phi_pp(:,i)/(20*10^-6)^2);
% OSPL = trapz(omega/(2*pi),Spp_db);
% Spp_db_A(:,i) = Spp_db(:,i) + A;
% OASPL = trapz(omega/(2*pi),Spp_db_A);

% new
Spp_db = 10*log10(S_pp(:,i)/(20*10^-6)^2);
Spp_db_A =  Spp_db + A;
spp_Pa = (20*10^-6)^2.*10.^(Spp_db_A/10);
Spp_A_Pa = trapz(omega/(2*pi),spp_Pa');  
OASPL = 10*log10(Spp_A_Pa(i)/(20*10^-6)^2); 

% figure(1)
% semilogx(omega/(2*pi), 10*log10(S_pp(:,i)/(20*10^-5)))
% hold on
% 
% figure(2)
% semilogx(omega/(2*pi), 10*log10(Phi_pp(:,i)/(20*10^-5)))
% hold on
end
end
end













