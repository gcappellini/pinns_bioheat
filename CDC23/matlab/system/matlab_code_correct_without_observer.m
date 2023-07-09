%From https://es.mathworks.com/help/matlab/math/solve-single-pde.html

clear all 
close all
clc

% Select Solution Mesh

x = 0:0.01:1; % 100 valori tra 0 e 1
t = 0:0.01:1; % 100 valori tra 0 e 1

% Solve Equation

m = 0;
sol = pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);

% Extract the first solution component from sol.

u = sol(:,:,1); % soluzione del sistema


% Plot Solution

surf(x,t,u)
title('Numerical solution computed with 100 mesh points')
xlabel('Distance x')
ylabel('Time t')



% Print Solution

fileID = fopen('output_matlab.txt','w');
%fprintf(fileID,'%6s %12s\n','x','exp(x)');
%fprintf(fileID,'%6.2f %12.8f %12.8f\n', t, x, u);

for i = 1:101
   for j = 1:101
        
     fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u(i,j));
        
   end
end



fclose(fileID);










% Code Equation

function [c,f,s] = pdex1pde(x,t,u,dudx)
omega=0.67;
c = 1;
f = dudx;
s = -u*omega;
end

% Code Initial Condition

function u0 = pdex1ic(x)
constant = 0.5;
u0 = constant*x;
end

% Code Boundary Conditions

function [pl,ql,pr,qr] = pdex1bc(xl,ul,xr,ur,t)
constant = 0.5;
pl = ul;
ql = 0;
pr = -constant;
qr = 1;
end



