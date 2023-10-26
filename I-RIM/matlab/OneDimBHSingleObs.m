%From https://es.mathworks.com/help/matlab/math/solve-single-pde.html

function [sol] = OneDimBHSingleObs
global normerr

m = 0;



x = 0:0.25:1; % 10 valori tra 0 e 1
t = 0:0.25:1; % 10 valori tra 0 e 1

sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x,t);
% % Extract the first solution component as u.  This is not necessary
% % for a single equation, but makes a point about the form of the output.
u1 = sol(:,:,1); %soluzione del sistema
u2 = sol(:,:,2); %soluzione dell'osservatore 1

% Print Solution PDE

fileID = fopen('output_matlab_pde_short.txt','w');
%fprintf(fileID,'%6s %12s\n','x','exp(x)');
%fprintf(fileID,'%6.2f %12.8f %12.8f\n', t, x, u);

for i = 1:5
   for j = 1:5
        
     fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u1(i,j));
        
   end
end

% Print solution observer

fileID = fopen('output_matlab_observer_short.txt','w');
%fprintf(fileID,'%6s %12s\n','x','exp(x)');
%fprintf(fileID,'%6.2f %12.8f %12.8f\n', t, x, u);

for i = 1:5
   for j = 1:5
        
     fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u2(i,j));
        
   end
end

% surface plot of the system solution
figure;
surf(x,t,u1);
title('Numerical solution of the system computed with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');
% surface plot of the observer solution 
figure;
surf(x,t,u2);
title('Numerical solution of the observer computed with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');

% surface plot of the observer solution 
figure;
surf(x,t,abs(u1-u2));
title('Observation error with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');

%solution profile at t_t_final
figure;
plot(x,u1(end,:),'o',x,u2(end,:),'x');

title('Solutions at t = t_{final}.');
legend('System','Observer1','Location', 'SouthWest');
xlabel('Distance x');
ylabel('temperature at t_final');
err=u1-u2;
for i=1:100
   normerr=[normerr;
   norm(err(i,:),2)];
end



%-----------------

% Code equation

function [c,f,s] = OneDimBHpde(x,t,u,dudx)
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
W_avg = 2.3;
c = [a1; a1];
f = [1; 1].* dudx;
s = [ -u(1)*a2*W_avg + a3*(qmet+beta*exp(-cc*L0*(X0-x))*p); 
     -u(2)*a2*W_avg + a3*(qmet+beta*exp(-cc*L0*(X0-x))*p); 
    ];

% --------------------------------------------------------------------------

% Code initial conditions

function u0 = OneDimBHic(x)
u0 = [0; (6/5 - x)*x];


% --------------------------------------------------------------------------

% Code boundary conditions

function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
k = 4;
pl = [ul(1); ul(2)];
ql = [0; 0];
pr = [1; 1-k*(ur(1)-ur(2))];
qr = [1;1];
