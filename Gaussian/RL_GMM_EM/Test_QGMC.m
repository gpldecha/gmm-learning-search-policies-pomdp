%% Test QGMC
clear all;
%% Load Q-value function
load('/home/guillaume/MatlabWorkSpace/RL/LeftOrRight/leftright_policy/Q_policy_greedy_c.mat');
Q       = Q_policy.Q;
%% Generate Data

a = -2.5;
b =  2.5;
nbSamples = 500;
Data = a + (b-a).*rand(2,nbSamples);

q = Q.f(Data');

%% Plot Value function
xs    = linspace(-2.5,2.5,100);
ys    = linspace(-2.5,2.5,100);
[X,Y] = meshgrid(xs,ys);
close all;
figure; hold on;
%pcolor(X,Y,reshape(q,size(X))); 
scatter(Data(1,:),Data(2,:),10,q);
%ghandle = plot_gmm_contour(gca,gmm.Priors,Mu_out,gmm.Sigma,[1 0 0],3);
ghandle = plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,[1 0 0],3);

xlabel('$\mathbf{x}$','Interpreter','Latex','FontSize',20);
ylabel('$\mathbf{\dot{x}}$','Interpreter','Latex','FontSize',20);
title(['Left right Control problem $\hat{Q}_{end}(\mathbf{x},\mathbf{\dot{x}})$'],'Interpreter','Latex','FontSize',15);
colorbar;
colormap('jet');
axis equal;

%% Uniform initial distribution
nbSamples   = 10;
x           = linspace(-2.5,2.5,nbSamples);
[X,Y]       = meshgrid(x,x);
K           = nbSamples;
D           = 2;
a           = -2.5;
b           =  2.5;
Mu          = zeros(2,nbSamples);
Mu(1,:)     = x;


Priors = ones(1,K) ./ K;
Sigma = [];
for k=1:K
    Sigma(:,:,k) = eye(D,D) .* (0.2)^2;
end

gmm.Priors = Priors;
gmm.Mu     = Mu;
gmm.Sigma  = Sigma;

%% Learn initial Policy
options = statset('Display','final','MaxIter',400);
GMModel = fitgmdist(Data',10,'Options',options,'CovarianceType','full','RegularizationValue',1e-05,'Replicates',5,'Start','plus');

gmm.Priors          = GMModel.ComponentProportion;
gmm.Mu              = GMModel.mu';
gmm.Sigma           = GMModel.Sigma;


%% Run QGMC_EM
in  = 1;
out = 2;
Mu_out = QGMC_EM(Data,q,gmm.Priors, gmm.Mu, gmm.Sigma,in,out);


Mu_out

%% Plot the conditional
%% Plot policy u = P(dx|x)

xs      = linspace(-2.5,2.5,50);
xs      = xs(:);

close all;
figure('units','normalized','position',[0.1 0.1 0.8 .4]) 

% ---------------------------- GMC ----------------------------------------

subplot(1,2,1);hold on; grid on; box on;
for i=1:size(xs,1)
    [Priors_c,Mu_c,Sigma_c] = GMC(Priors, Mu_out, Sigma, xs(i), 1, 2);
    w   = rescale(Priors_c,0,max(Priors_c),0,1);
    idx = find(w >= 0.1);
    idx = idx(:);
    
    pts = repmat(xs(i),size(idx,1),1);
    dir = Mu_c(idx);
    
    w_s = w(idx);
    
    w_s = rescale(w_s,0.1,1,5,50);
    
    scatter(pts,dir,w_s,'k');   
    
end

plot([3,-3],[0,0],'--k','LineWidth',1);

xlabel('$\mathbf{x}$','Interpreter','Latex','FontSize',20);
ylabel('$\mathbf{\dot{x}}$','Interpreter','Latex','FontSize',20);
title('GMM Modes');
set(gca,'FontSize',12);

% ---------------------------- GMR ----------------------------------------

subplot(1,2,2);hold on; grid on; box on;

[y, Sigma_y] = GMR(Priors, Mu_out, Sigma, xs', 1, 2);
Sigma_y      = squeeze(Sigma_y);
ws           = rescale(Sigma_y,min(Sigma_y),max(Sigma_y),5,50);
scatter(xs,y,ws,'k');   

idx1 = find(y <= 0);
idx2 = find(y >= 0);

hs(1) = scatter(xs(idx1),zeros(size(idx1,2),1),20,[1 0 0],'filled');
hs(2) = scatter(xs(idx2),zeros(size(idx2,2),1),20,[0 0 1],'filled');

plot([3,-3],[0,0],'--k','LineWidth',1);

legend(hs,'left','right','Location','SouthWest');

xlabel('$\mathbf{x}$','Interpreter','Latex','FontSize',20);
ylabel('$\mathbf{\dot{x}}$','Interpreter','Latex','FontSize',20);
title('GMR');
ylim([-2 2]);
set(gca,'FontSize',12);





