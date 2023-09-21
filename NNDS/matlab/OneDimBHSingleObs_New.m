%From https://es.mathworks.com/help/matlab/math/solve-single-pde.html

function [sol] = OneDimBHSingleObs

m = 0;



x = 0:0.01:1; % 100 valori tra 0 e 1
t = 0:0.01:1; % 100 valori tra 0 e 1
w = 0.4:0.4:4; % 10 valori tra 0.4 e 4

sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x,t);
% % Extract the first solution component as u.  This is not necessary
% % for a single equation, but makes a point about the form of the output.
u1 = sol(:,:,1); %soluzione del sistema

% Print Solution PDE

fileID = fopen('output_matlab_pde.txt','w');

for i = 1:101
   for j = 1:101
       for k = 1:11
        
        fprintf(fileID,'%6.2f %6.2f %12.8f %12.8f %6.2f\n', x(j), t(i), u1(i,101), u1(i,j), w(k));
        
   
       end
   end
end



%-----------------

% Code equation

function [c,f,s] = OneDimBHpde(x,t,w,u,dudx)
%La prima equazione è quella del sistema, a seguire gli osservatori
a1 = 1.061375;
a2 = 1.9125;
a3 = 6.25e-05;
qmet = 4200;
beta = 1;
cc = 16;
L0 = 0.05;
X0 = 0.09;
p = 150/(1.97e-3);
c = a1;
f = dudx;
s = -u(1)*a2*w + a3*(qmet+beta*exp(-cc*L0*(X0-x))*p);

% --------------------------------------------------------------------------

% Code initial conditions

function u0 = OneDimBHic(x)
u0 = 0;


% --------------------------------------------------------------------------

% Code boundary conditions

function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
pl = ul(1);
ql = 0;
pr = 1;
qr = 1;
