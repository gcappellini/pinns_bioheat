%From https://es.mathworks.com/help/matlab/math/solve-single-pde.html

function [sol] = OneDimBHSingleObs
global k normerr omega

omega = 0.67;

k = 4;

m = 0;



x = 0:0.01:1; % 100 valori tra 0 e 1
t = 0:0.01:1; % 100 valori tra 0 e 1

sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x,t);
% % Extract the first solution component as u.  This is not necessary
% % for a single equation, but makes a point about the form of the output.
u1 = sol(:,:,1); %soluzione del sistema
u2 = sol(:,:,2); %soluzione dell'osservatore 1

% Print Solution PDE

fileID = fopen('output_matlab_pde.txt','w');
%fprintf(fileID,'%6s %12s\n','x','exp(x)');
%fprintf(fileID,'%6.2f %12.8f %12.8f\n', t, x, u);

for i = 1:101
   for j = 1:101
        
     fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u1(i,j));
        
   end
end

% Print solution observer

fileID = fopen('output_matlab_observer.txt','w');
%fprintf(fileID,'%6s %12s\n','x','exp(x)');
%fprintf(fileID,'%6.2f %12.8f %12.8f\n', t, x, u);

for i = 1:101
   for j = 1:101
        
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
surf(x,t,u1-u2);
title('Observation error with 100 mesh points.');
xlabel('Distance x');
ylabel('Time t');

%solution profile at t_t_final
figure;
plot(x,u1(end,:),'o',x,u2(end,:),'x');

title('Solutions at t = t_final.');
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
W_avg = 2.3;
c = [a1; a1];
f = [1; 1].* dudx;
s = [-u(1)*a2*W_avg; 
     -u(2)*a2*W_avg; 
    ];

% --------------------------------------------------------------------------

% Code initial conditions

function u0 = OneDimBHic(x)
q0_ad = 3.125;
u0 = [(q0_ad/2)*x^2; (q0_ad)*x];


% --------------------------------------------------------------------------

% Code boundary conditions

function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
k = 4;
q0_ad = 3.125;
pl = [ul(1); ul(2)];
ql = [0; 0];
pr = [-q0_ad; -q0_ad-k*(ur(1)-ur(2))];
qr = [1;1];
