function [Strip,fluid,input] = inputs_definition_StripTheory()

Strip = struct; 
Strip.n = 28;
prompt         = {'Number of sections:'};
dlg_title      = 'Input';
num_lines      = 1;
defaultans     = {num2str(Strip.n)};
data           = inputdlg(prompt,dlg_title,num_lines,defaultans);
Strip.n        = str2double(data{1});

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
input.chord     = 0.4;
input.span      = 0.5;
input.U         = 56;
input.x         = 0;
input.y         = 0;
input.z         = 1;
input.AoA       = 0;
input.xtr_s     = 0.06;
input.xtr_p     = 0.06;
input.Mic       =0.99;

prompt         = {'Perfil:', 'Chord [m]:', 'Span [m]:','U$_\infty$ [m/s]:'...
, 'Angle of Attack:', 'Transition suction side [x/c]', 'Transition presusre side [x/c]:',...
'Position surface mic TE [x/c]:' 'Mic position x [m]:', 'Mic position y [m]:',...
'Mic position z [m]:','K_y'};
dlg_title      = 'Input';
num_lines      = 1;
defaultans     = {num2str(input.perfil),num2str(input.chord),num2str(input.span)...
 ,num2str(input.U),num2str(input.AoA),num2str(input.xtr_s),num2str(input.xtr_p),...
 num2str(input.Mic),num2str(input.x),num2str(input.y),num2str(input.z),'none'};
data           = inputdlg(prompt,dlg_title,num_lines,defaultans);

input.perfil   = data{1};
input.chord    = str2double(data{2});
input.span_total = str2double(data{3});
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
[indx] = listdlg('ListString', list, 'Name', 'Surface pressure model');
input.model = list(indx);
end 