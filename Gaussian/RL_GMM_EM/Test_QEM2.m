%% Test QEM with Advantage function
%% Load Q-value function
load('/home/guillaume/MatlabWorkSpace/RL/LeftOrRight/leftright_policy/Q_policy_greedy_c.mat');
Q       = Q_policy.Q;
%% Plot Value function
xs    = linspace(-2.8,2.8,100);
ys    = linspace(-2.8,2.8,100);
[X,Y] = meshgrid(xs,ys);

qs = Q.f([X(:),Y(:)]);
qs = reshape(qs,size(X));
v  = sum(qs)./55;

options_v.var = (0.5).^2;
V = LLR(options_v);
V.train(xs',v');

%% Load data
Data = loadAllMatFiles('/home/guillaume/MatlabWorkSpace/RL/LeftOrRight/Data/record',true);
xurxp = create_tuples(Data);
clear Data
F     = [];
for i=1:size(xurxp,1)
     F   = [F;xurxp{i}.F];
end
mu_F = mean(F);
sd_F = std(F);
F     = [];
FP    = [];
U     = [];
R     = [];
flags = [];
A     = [];
% A(x,u) = r(x,u) + \gamma * V(xp) - V(x)
disc = 0.9;
Data = [];
for i=1:size(xurxp,1)
    F       = (xurxp{i}.F - mu_F)./sd_F;
    vf      = V.f(F);
    
    FP      = (xurxp{i}.FP - mu_F)./sd_F;
    vfp     = V.f(FP);
    
    U       = xurxp{i}.U;
    R       = xurxp{i}.R;
    
    A       = R + disc * vfp - vf;
    A(end) = R(end);
    
    Data = [Data,[F';U';A']];
    
end
%clear xurxp;
clear i;
%% Plot Q(x,u) and V(x) (ground truth)
xs    = linspace(-2.8,2.8,100);
ys    = linspace(-2.8,2.8,100);
[X,Y] = meshgrid(xs,ys);

%close all;
figure; 
subplot(1,2,1);
hold on;
pcolor(X,Y,qs);shading interp;
xlabel('$\mathbf{x}$','Interpreter','Latex','FontSize',20);
ylabel('$\mathbf{\dot{x}}$','Interpreter','Latex','FontSize',20);
title(['Left right Control problem $\hat{Q}_{end}(\mathbf{x},\mathbf{\dot{x}})$'],'Interpreter','Latex','FontSize',15);
colorbar;
colormap('jet');
subplot(1,2,2);
plot(xs,V.f(xs'));
xlabel('$\mathbf{x}$','Interpreter','Latex','FontSize',20);
ylabel('$V(\mathbf{x})$','Interpreter','Latex','FontSize',20);
title(['Left right Control problem $\hat{V}_{end}(\mathbf{x})$'],'Interpreter','Latex','FontSize',15);

%% Generate Gaussians

a = -2;
b =  2;
nbSamples = 5;
Priors = ones(1,nbSamples)./nbSamples;
Mu     = a + (b-a).*rand(2,nbSamples);
Sigma = repmat(eye(2,2).*(0.5.^2),[1,1,nbSamples]);

%% Plot Advantage function 
Av = rescale(Data(3,:),min(Data(3,:)),max(Data(3,:)),0,100);
Av = Av(:);

close all;
figure; hold on;
scatter(Data(1,:),Data(2,:),15,Av);
ghandle = plot_gmm_contour(gca,Priors,Mu,Sigma,[1 0 0],3);
colormap('jet');
colorbar;

%% Run QEM

[Priors, Mu, Sigma, Pix] = QEM(Data(1:2,:),Av,Priors, Mu, Sigma ,'full');

%% Plot the conditional
%% Plot policy u = P(dx|x)

xs      = linspace(-2.5,2.5,50);
xs      = xs(:);

close all;
figure('units','normalized','position',[0.1 0.1 0.8 .4])

% ---------------------------- GMC ----------------------------------------

subplot(1,2,1);hold on; grid on; box on;
for i=1:size(xs,1)
    [Priors_c,Mu_c,Sigma_c] = GMC(Priors, Mu, Sigma, xs(i), 1, 2);
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

[y, Sigma_y] = GMR(Priors, Mu, Sigma, xs', 1, 2);
Sigma_y      = squeeze(Sigma_y);
if length(Priors == 1)
    ws = 25;
else
    ws = rescale(Sigma_y,min(Sigma_y),max(Sigma_y),5,50);
end

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





