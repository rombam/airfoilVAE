function [Phi_pp_s,Phi_pp_p,Cl,Cd, Cdp, Cm, Cf, valid_res] = surface_pressure_spectrum(omega,inputs,fluid,model)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

%% inputs from XFOIL
% [delta_s_s,delta_s_p,theta_s,theta_p,cf_s,cf_p,x_c,Cp,Ue_s,Ue_p,dcpdx_s,dcpdx_p, Cl, Cd, Cdp, Cm, Cf] = XFOIL(inputs.perfil,inputs.AoA,inputs.Re,inputs.M,inputs.xtr_s,inputs.xtr_p,inputs.chord,inputs.Mic);
[delta_s_s,delta_s_p,theta_s,theta_p,cf_s,cf_p,x_c,Cp,Ue_s,Ue_p,dcpdx_s,dcpdx_p, Cl, Cd, Cdp, Cm, Cf, valid_res] = XFOIL_new_airfoil(inputs.perfil,inputs.AoA,inputs.Re,inputs.M,inputs.xtr_s,inputs.xtr_p,inputs.chord,inputs.Mic,'shape.dat');

if valid_res == 1
%% boundary layer parameter for each side of the airfoil
[delta_s,tau_w_s,u_t_s] = Boundary_layer_characteristics(delta_s_s,theta_s,cf_s,Ue_s,inputs,fluid);
[delta_p,tau_w_p,u_t_p] = Boundary_layer_characteristics(delta_s_p,theta_p,cf_p,Ue_p,inputs,fluid);

%% for validation purposes - uncomment:
% Ue_s = 0.9186;
% Ue_p = 0.9186;
% delta_s = 0.4*0.0229;
% delta_p = 0.4*0.0229;
% delta_s_s = 0.4*0.0057;
% delta_s_p = 0.4*0.0057;
% theta_s = 0.4*0.0034;
% theta_p = 0.4*0.0034;
% u_t_s = inputs.U*0.0277;
% u_t_p = inputs.U*0.0277;
% tau_w_s = fluid.rho*u_t_s^2;
% tau_w_p = fluid.rho*u_t_p^2;

%% Several wall pressure models 
tf = strcmp(model,'Goody');
if tf == 1   
    [a_s,b_s,c_s,d_s,e_s,f_s,g_s,h_s,i_s,R_s,phi_s_s,omega_s] = Goody(omega,tau_w_s,delta_s,Ue_s,u_t_s,fluid,inputs);
    [Phi_pp_s] = phi_pp_oneside(a_s,b_s,c_s,d_s,e_s,f_s,g_s,h_s,i_s,R_s,phi_s_s,omega_s);
    [a_p,b_p,c_p,d_p,e_p,f_p,g_p,h_p,i_p,R_p,phi_s_p,omega_s] = Goody(omega,tau_w_p,delta_p,Ue_p,u_t_p,fluid,inputs);
    [Phi_pp_p] = phi_pp_oneside(a_p,b_p,c_p,d_p,e_p,f_p,g_p,h_p,i_p,R_p,phi_s_p,omega_s);
end
tf = strcmp(model,'Kamruzzaman');
if tf ==1
    [a_s,b_s,c_s,d_s,e_s,f_s,g_s,h_s,i_s,R_s,phi_s_s,omega_s] = Kamruzzaman(inputs,fluid,cf_s, theta_s, dcpdx_s,delta_s_s,Ue_s, omega,u_t_s);
    [Phi_pp_s] = phi_pp_oneside(a_s,b_s,c_s,d_s,e_s,f_s,g_s,h_s,i_s,R_s,phi_s_s,omega_s);
    [a_p,b_p,c_p,d_p,e_p,f_p,g_p,h_p,i_p,R_p,phi_s_p,omega_s] = Kamruzzaman(inputs,fluid,cf_p, theta_p, dcpdx_p,delta_s_p,Ue_p, omega,u_t_s);
    [Phi_pp_p] = phi_pp_oneside(a_p,b_p,c_p,d_p,e_p,f_p,g_p,h_p,i_p,R_p,phi_s_p,omega_s);    
end 
tf = strcmp(model,'TNO');
if tf ==1
    [Phi_pp_s] = TNO(delta_s,u_t_s,fluid,inputs, dcpdx_s,tau_w_s,delta_s_s,omega);
    [Phi_pp_p] = TNO(delta_p,u_t_p,fluid,inputs, dcpdx_p,tau_w_p,delta_s_p,omega);
end
tf = strcmp(model,'Amiet');
if tf == 1
    [Phi_pp_s] = Amiet_Sqq(inputs, fluid, omega, delta_s_s);
    [Phi_pp_p] = Amiet_Sqq(inputs, fluid, omega, delta_s_p);
end

else

    Phi_pp_s = 0;Phi_pp_p = 0;Cl = 0;Cd = 0;Cdp = 0;Cm = 0;Cf = 0;
end
end

