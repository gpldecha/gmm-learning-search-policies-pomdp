%% Gaussian Mixutre Conditional
%% Taking the Expectation in a multi-model distribution over the velocity
%% is a bad idea. Need a way to select velocities.

W      = [[-1,0];[0,1];[0,-1];[1,0]];
pi     = ones(1,4)./4;
w      = pi;% pi/max(pi);
colors = jet(200);

%% Plot Velocity modes
close all;
figure; hold on; box on; axis on;
daspect manual
pbaspect manual

set(gcf,'color','w');
set(0,'defaulttextinterpreter','latex');
axis([-1 1 -1 1]);

global ColorOrder; ColorOrder=[];
set(gca,'ColorOrder',colors(round(rescale(w,0,1,1,size(colors,1)))',:));
arrow3(zeros(4,2),W,'o',[],[],1);
colormap(colors);
colorbar;

set(gca,'FontSize',12);
title('Velocity modes');
xlabel('x');
ylabel('y');

%% test cases

vtmp = [0,1];

pis = [1,  1,  1, 1;
    0.2, 0.2, 0.2, 0.4;
    0.1, 0.1, 0.1, 0.7;
    1, 0.01, 0.01, 0.01
    ];
pis = normr(pis);

close all;
figure; hold on; box on; axis on;
daspect manual
pbaspect manual

set(gcf,'color','w');
set(0,'defaulttextinterpreter','latex');
axis([-1 1 -1 1]);

for i=1:size(pis);
    subplot(2,2,i);
    
    w = pis(i,:);
     
    [v,alphas] = gmc_velocity(w,W,vtmp);
    alphas
    %ARROW3(P1,P2,S,W,H,IP,ALPHA,BETA)
    set(gca,'ColorOrder',colors(round(rescale(w,0,1,1,size(colors,1)))',:));
    arrow3(zeros(4,2),W,'o',[],[],1);    hold off;
    hold on;
    arrow3(zeros(1,2),vtmp,'r2',5,5);
    arrow3(zeros(1,2),v,'k2',5,5);
    hold off;
    colormap(colors);
    colorbar;
end


