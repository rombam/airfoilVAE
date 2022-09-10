function [omega, S_pp,Phi_pp_s, Cl, Cd, Cdp, Cm, Cf, valid_res] = TE_noise_Prediction(inputs,fluid)

%% Define waves number
[f,omega,B,C,K_bar,k_min_bar,mu_bar,S0,K_1_bar,alpha,U_c,K_2_bar] = wavesnumbers(inputs,fluid);

%% transfer function f1 (identical to Amiet)
[f1] = Radiation_integral1(B,C);

%% Transfer function f2 (for the Back-scattering effect)
[f2] = Raditation_integral2(B,K_bar,k_min_bar,mu_bar,S0,K_1_bar,alpha,inputs);

%% Radiation integral total
[I] = Radiation_integral(f1,f2);

%% Spanwise correlation length 
[l_y] = spanwise_corlength(U_c,omega,K_2_bar,inputs);

%% Wall pressure PSD
[Phi_pp_s,Phi_pp_p,Cl, Cd, Cdp, Cm,Cf ,valid_res] = surface_pressure_spectrum(omega,inputs,fluid,inputs.model);

if valid_res == 1
%% Stream-wise integrated wavenumber spectral density of wall-pressure fluctuations
[Pi0_s,Pi0_p] = Integrated_WPS(l_y,Phi_pp_s,Phi_pp_p);

%% far-field noise
[S_pp_s,S_pp_p] = farfield_noise(inputs,fluid,omega,S0,I,Pi0_s,Pi0_p);
S_pp = S_pp_s + S_pp_p;

%% plots
%plots(f,I,Phi_pp_s,Phi_pp_p,S_pp,S_pp_s,S_pp_p)
else
    S_pp = 0;
end
end

