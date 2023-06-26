%From https://es.mathworks.com/help/matlab/math/solve-single-pde.html

function [sol] = syst
m = 0;

x = 0:0.01:1; % 100 valori tra 0 e 1
t = 0:0.01:1; % 100 valori tra 0 e 1

sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x,t);
% % Extract the first solution component as u.  This is not necessary
% % for a single equation, but makes a point about the form of the output.
u1 = sol(:,:,1); %soluzione del sistema

% Print Solution PDE

fileID = fopen('output_matlab_system_1.txt','w');
%fprintf(fileID,'%6s %12s\n','x','exp(x)');
%fprintf(fileID,'%6.2f %12.8f %12.8f\n', t, x, u);

for i = 1:101
   for j = 1:101
        
     fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u1(i,j));
        
   end
end


% surface plot of the system solution
figure;
surf(x,t,u1);
title('Numerical solution of the system computed with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');


%-----------------

% Code equation

function [c,f,s] = OneDimBHpde(x,t,u,dudx)
%La prima equazione è quella del sistema, a seguire gli osservatori
a1 = 1.061375;
a2 = 1.9125;
W_avg = 2.3;
c = a1;
f = 1.* dudx;
s = -u(1)*a2*W_avg;

% --------------------------------------------------------------------------

% Code initial conditions

function u0 = OneDimBHic(x)
q0_ad = 2;
b = 1.875;
u0 = (q0_ad/4)*x^4 + b*x*(x-1)^2;


% --------------------------------------------------------------------------

% Code boundary conditions

function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
q0_ad = 2;
pl = ul(1);
ql = 0;
%pr = -q0_ad; %constant flux
pr = -q0_ad*(1-t); %linear flux
%pr = -q0_ad*exp(-t); %exponential flux
%pr = -q0_ad*cos(pi*t/2); %sinusoidal flux
qr = 1;
