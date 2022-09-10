function [fluid,input] = inputs_definition()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here



%% fluid properties 
fluid           = struct; 
fluid.T         = 22;
fluid.Tk        = fluid.T+273.15;
fluid.P         = 100000;
fluid.ni        = 1.5e-5;
fluid.gamma     = 1.4;
fluid.R         = 287;
fluid.c0        = sqrt(fluid.gamma*fluid.R*fluid.Tk);
fluid.rho       = 1.181;
%fluid.rho       = fluid.P/(fluid.R*fluid.Tk);

%% test case parameters
%default ans:
input           = struct;
input.perfil    = 'Naca0012';
input.chord     = 1;
input.span      = 1;
input.U         = 56;
input.x         = 0;
input.y         = 0;
input.z         = 1;
input.AoA       = 3;
input.xtr_s     = 0.06;
input.xtr_p     = 0.06;
input.Mic       =0.99;

prompt         = {'Airfoil:', 'Chord [m]:', 'Span [m]:','U_inf [m/s]:'...
, 'Angle of Attack [degree]:', 'Transition suction side [x/c] (1 for NT):', 'Transition presusre side [x/c] (1 for NT):',...
'Position surface mic TE [x/c]:' 'Mic position x [m]:', 'Mic position y [m]:',...
'Mic position z [m]:','K_y'};
dlg_title      = 'Input';
num_lines      = 1;
defaultans     = {num2str(input.perfil),num2str(input.chord),num2str(input.span)...
 ,num2str(input.U),num2str(input.AoA),num2str(input.xtr_s),num2str(input.xtr_p),...
 num2str(input.Mic),num2str(input.x),num2str(input.y),num2str(input.z),'none'};
% data           = inputdlg(prompt,dlg_title,num_lines,defaultans);
data           = defaultans;

input.perfil   = data{1};
input.chord    = str2double(data{2});
input.span     = str2double(data{3});
input.semichord = input.chord/2;
input.U        = str2double(data{4});
input.AoA      = str2double(data{5});
input.xtr_s    = str2double(data{6});
input.xtr_p    = str2double(data{7});
input.Mic      = str2double(data{8});
input.x        = str2double(data{9});
input.y        = str2double(data{10});
input.z        = str2double(data{11});
input.Ky       = NaN;
input.M        = input.U/fluid.c0;
input.Re       = input.chord*input.U/fluid.ni;
list = {'Goody','Kamruzzaman','Amiet','TNO'};
%[indx] = listdlg('ListString', list, 'Name', 'Surface pressure model');
indx = 4;
input.model = list(indx);

%% filter:
%% OASPL: Calculate the A filter.
f = 100:20000;
Ra = 12194^2*f.^4./((f.^2+20.6^2).*sqrt((f.^2+107.7^2).*(f.^2+797.9^2)).*(f.^2+12194^2));
input.A = 20*log10(Ra)-20*log10(Ra(f==1000));
end

