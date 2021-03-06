%% Load the components of the GMM model which are on the table

foldername  = './gmm-learning-search-policies-pomdp';

path_to_gmm = [foldername '/model/model_surface/'];
Priors      = load([path_to_gmm 'pi.mat'],'-ascii');
Mu          = load([path_to_gmm 'mu.mat'],'-ascii');
D           = size(Mu,1);
K           = size(Mu,2);
Sigma       = load([path_to_gmm 'sigma.mat'],'-ascii');
Sigma       = reshape(Sigma,D,D,K);


%% Plot GMM + Table + Axis

close all;
hf = figure; hold on;grid on;

[NewMu NewSigma]    = getGaussianSlice(Mu,Sigma,[4 5 6]);
handles             = plot3dGaussian(Priors, NewMu,NewSigma );
alpha               = rescale(Priors,min(Priors),max(Priors),0.1,0.8);

for i=1:size(handles,1)
   set(handles(i),'FaceLighting','phong','FaceColor',[0.5,0.5,0.5],'FaceAlpha',alpha(i),'AmbientStrength',0.1,'EdgeColor','none');
end

plotcube([0.5 0.7 0.05],[ -0.25 -0.35  -0.025],1,[1 1 1]);
plotcube([0.03  0.06 0.025],[ 0.15 0.125 0.025],1,[0.0 1.0 0.0]);

set(gca,'ZTick',[-0.02 0.08]);
camlight

axis equal;
hold off;

title('Gaussian Mixture Model (strategies)','FontSize',16);
set(gca,'FontSize',16);






