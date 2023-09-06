
function [sol] = OneDimBH
global k lambda flusso om1 om2  om3 om4   om5 om6 om7 om8 


err1=0
rms1=[0]
m = 0;
x = linspace(0,1,100);
t = linspace(0,600,100);

sol = pdepe(m,@OneDimBHpde,@OneDimBHic,@OneDimBHbc,x,t);
% % Extract the first solution component as u.  This is not necessary
% % for a single equation, but makes a point about the form of the output.
u1 = sol(:,:,1); %soluzione del sistema
u2 = sol(:,:,2); %soluzione dell'osservatore 1
u3 = sol(:,:,3); %soluzione dell'osservatore 2
u4 = sol(:,:,4); %soluzione dell'osservatore 3
u5 = sol(:,:,5); %soluzione dell'osservatore 4
u6 = sol(:,:,6); %soluzione dell'osservatore 5
u7 = sol(:,:,7); %soluzione dell'osservatore 6
u8 = sol(:,:,8); %soluzione dell'osservatore 7
u9 = sol(:,:,9); %soluzione dell'osservatore 8

u10 = sol(:,:,10); %soluzione del peso 1
u11 = sol(:,:,11); %soluzione del peso 2
u12 = sol(:,:,12); %soluzione del peso 3
u13 = sol(:,:,13); %soluzione del peso 4
u14 = sol(:,:,14); %soluzione del peso 5
u15 = sol(:,:,15); %soluzione del peso 6
u16 = sol(:,:,16); %soluzione del peso 7
u17 = sol(:,:,17); %soluzione del peso 8

figure
plot(t,u10,'r',t,u11,'g',t,u12,'b',t,u13,'yellow',t,u14,'cyan',t,u15,'-.',t,u16,'--',t,u17,'black') %plot the dynamic wheights
title('dynamic wheights');

%multiple-model temperature estimation
uav=u2.*u10+u3.*u11+u4.*u12+u5.*u13+u6.*u14+u7.*u15+u8.*u16+u9.*u17;

% % surface plot of the system solution
% figure;
% surf(x,t,u1);
% title('Numerical solution of the system computed with 20 mesh points.');
% xlabel('Distance x');
% ylabel('Time t');
% % surface plot of the observer solution 
% figure;
% surf(x,t,uav);
% title('Numerical solution of the observer computed with 20 mesh points.');
% xlabel('Distance x');
% ylabel('Time t');
% % surface plot of the observer solution 2
% figure;
% surf(x,t,u3);
% title('Numerical solution of the observer computed with 20 mesh points.');
% xlabel('Distance x');
% ylabel('Time t');


% figure
% plot(t,err1,'r',t,err2,'g',t,err3,'b',t,err4,'cyan')

% % surface plot of the system solution
% figure;
% surf(x,t,uav);
% title('Numerical solution of the system computed with 20 mesh points.');
% xlabel('Distance x');
% ylabel('Time t');

%solution profile at t_t_final
figure;
plot(x,u1(end,:),'o',x,u2(end,:),'r',x,u3(end,:),'g',x,u4(end,:),'b',x,u5(end,:),'cyan',...
    x,u6(end,:),'.',x,u7(end,:),'-.',x,u8(end,:),'*',x,u9(end,:),'yellow',x,uav(end,:),'x');

title('Solutions at t = t_{final}.');
legend('System','Observer1','Observer2','Observer3','Observer4','Observer5',...
    'Observer6','Observer7','Observer8','ObserverMultiModel','Location', 'SouthWest');
xlabel('Distance x');
ylabel('temperature at t_{final}');


%-----------------
function [c,f,s] = OneDimBHpde(x,t,u,dudx)
global lambda om1 om2 om3 om4 om5 om6 om7 om8 
%la prima equazione è quella del sistema, a seguire gli osservatori
t

c = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1];
f = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1].* dudx;

den=u(10)*exp(-om1)+u(11)*exp(-om2)+u(12)*exp(-om3)+u(13)*exp(-om4)+...
    u(14)*exp(-om5)+u(15)*exp(-om6)+u(16)*exp(-om7)+u(17)*exp(-om8);

s = [-0.67*u(1); 
    -0.67*u(2); 
    -20*u(3); 
    -1*u(4); 
    -14*u(5); 
    -9*u(6); 
    -8*u(7); 
    -7*u(8); 
    -6*u(9); 
    -lambda*u(10)*(1-(exp(-om1)/den));
    -lambda*u(11)*(1-(exp(-om2)/den)); 
    -lambda*u(12)*(1-(exp(-om3)/den)); 
    -lambda*u(13)*(1-(exp(-om4)/den));
    -lambda*u(14)*(1-(exp(-om5)/den));
    -lambda*u(15)*(1-(exp(-om6)/den)); 
    -lambda*u(16)*(1-(exp(-om7)/den)); 
    -lambda*u(17)*(1-(exp(-om8)/den));
    ];






% --------------------------------------------------------------------------

function u0 = OneDimBHic(x)
global k flusso 
u0 = [flusso*x; flusso*x;  flusso*x; flusso*x; flusso*x; flusso*x; flusso*x; flusso*x; flusso*x; 0.2; 0.1; 0.1; 0.1; 0.1; 0.2; 0.1; 0.1];

% --------------------------------------------------------------------------

function [pl,ql,pr,qr] = OneDimBHbc(xl,ul,xr,ur,t)
global k flusso om1 om2 om3 om4 om5 om6 om7 om8 
pl = [ul(1);ul(2);ul(3);ul(4);ul(5);ul(6);ul(7);ul(8);ul(9);0;0;0;0;0;0;0;0];
ql = [0;0;0;0;0;0;0;0;0;1;1;1;1;1;1;1;1];
pr = [flusso;
    flusso-k*(ur(1)-ur(2));
    flusso-k*(ur(1)-ur(3));
    flusso-k*(ur(1)-ur(4));
    flusso-k*(ur(1)-ur(5));
    flusso-k*(ur(1)-ur(6));
    flusso-k*(ur(1)-ur(7));
    flusso-k*(ur(1)-ur(8));
    flusso-k*(ur(1)-ur(9));
    0;0;0;0;0;0;0;0]; %flusso negativo, con osservatore
qr = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
om1=0.5*((pr(2)-pr(1))/k)^2;
om2=0.5*((pr(3)-pr(1))/k)^2;
om3=0.5*((pr(4)-pr(1))/k)^2;
om4=0.5*((pr(5)-pr(1))/k)^2;
om5=0.5*((pr(6)-pr(1))/k)^2;
om6=0.5*((pr(7)-pr(1))/k)^2;
om7=0.5*((pr(8)-pr(1))/k)^2;
om8=0.5*((pr(9)-pr(1))/k)^2;


