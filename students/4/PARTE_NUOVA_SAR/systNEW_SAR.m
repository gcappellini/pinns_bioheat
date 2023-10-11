
function [sol] = syst
m = 0;

x = 0:0.01:1; % 100 valori tra 0 e 1
t = 0:0.01:1; % 100 valori tra 0 e 1


sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x,t);
% % Extract the first solution component as u.  This is not necessary
% % for a single equation, but makes a point about the form of the output.
u1 = sol(:,:,1); %soluzione del sistema

% %% STAMPA
u = round(u1,4);
l = zeros(1,length(u));
writematrix(l,'RISULTATI_SAR.txt')
writematrix(u,'RISULTATI_SAR.txt','WriteMode','append')


% surface plot of the system solution
figure;
surf(x,t,u1);
title('Numerical solution of the system computed with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');

% Code equation

function [c,f,s] = OneDimBHpde(x,t,u,dudx)
%La prima equazione è quella del sistema, a seguire gli osservatori
a1 = 1.0614;
a2 =1.9125;
a3 = 6.25e-05;
qmet = 4200;
p = 150/(1.97e-3);
W_avg = 2.3;
c = a1;
f = 1.* dudx;
s = -u(1)*a2*W_avg+ a3*(qmet+exp(-16*0.05*(0.09-x))*p);

% --------------------------------------------------------------------------

% Code initial conditions

function u0 = OneDimBHic(x)
u0 = ((16/8)*x^4)/4 + (15/8)*x*((x-1)^2);


% --------------------------------------------------------------------------

% Code boundary conditions

function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
pl = ul(1);
ql = 0;
pr = 1; %Dirichlet
qr = 1;
