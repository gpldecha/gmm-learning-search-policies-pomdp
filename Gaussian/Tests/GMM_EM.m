%% Test EM with GMM
%% Generate GMM from data

Priors       = [0.2,0.4,0.4];
Mu           = [];
Mu(:,1)      = [-2 0]';
Mu(:,2)      = [2 1]';
Mu(:,3)      = [5 0]';
Sigma        = [];
Sigma(:,:,1) = [3,0;0,0.1];
Sigma(:,:,2) = [2,1;1,1.1];
Sigma(:,:,3) = [1,0;0,1];

for i=1:3
    Sigma(:,:,i) =  Sigma(:,:,i) + eye(2,2) .* 1e-03;
    % Sigma(:,:,i) = 0.5 .* ( Sigma(:,:,i)  +  Sigma(:,:,i)');
end

XTrain = gmm_sample(500,Priors,Mu,Sigma)';


%% Plot 2D gaussian (in 2D)

close all;
figure;
hold on; grid on;
plot_gmm_contour(gca,Priors,Mu,Sigma,[0 0 1]);
plot(XTrain(:,1),XTrain(:,2),'+r');
axis equal;

%% EM Sylvain

[N,D]               = size(XTrain);
CovarianceTpye      = 'full';

if strcmp(CovarianceTpye,'full')
    nbParamK=1+D+D*(1+D)/2;
elseif strcmp(CovarianceTpye,'diagonal')
    nbParamK=1+D+D;
elseif strcmp(CovarianceTpye,'isotropic')
    nbParamK=1+D+1;
end

K = 10;
bics   = zeros(K,1);
loglik = zeros(K,1);

tic
for k=1:K
    
    [Priors_i, Mu_i, Sigma_i]  = EM_init_kmeans(XTrain', k,CovarianceTpye);
    [Priors_j, Mu_j, Sigma_j]  = EM(XTrain',Priors_i, Mu_i, Sigma_i,CovarianceTpye);
    
     loglik(k)                  = LogLikelihood_gmm(XTrain,Priors_j,Mu_j,Sigma_j);
     bic                        = BIC_f(loglik(k),k,nbParamK,N);
     bics(k)                    = bic;
%     disp(['k: ' num2str(k) ' loglik: ' num2str(loglik(k)) ' bic: ' num2str(bic)]);
    
end
toc

%% EM Matlab

options = statset('Display','off','MaxIter',400);
K       = 10;
bics     = zeros(K,1);
AIC     = zeros(K,1);



tic;
for k = 1:K
    %disp(['k(' num2str(k) ')']);
    GMModel = fitgmdist(XTrain,k,'Options',options,'CovarianceType','full','RegularizationValue',1e-04,'Replicates',1,'Start','plus');
    bics(k)  = GMModel.BIC;
    AIC(k)  = GMModel.AIC;
end
toc



%% Plot BIC curve
%close all;
figure; hold on; box on;
subplot(1,2,1);
plot(bics);
subplot(1,2,2);
plot(loglik);


%% disp(['k: ' num2str(k) ' loglik: ' num2str(loglik) 'bic: ' num2str(BIC(1,k))]);



k                              = 3;

[Priors_i, Mu_i, Sigma_i]      = EM_init_kmeans(XTrain', k,CovarianceTpye);
[Priors_j, Mu_j, Sigma_j, Pix] = EM(XTrain',Priors_i, Mu_i, Sigma_i,CovarianceTpye);

%% Plot 2D gaussian (in 2D)

close all;
figure;
subplot(1,2,1);
hold on; grid on;
plot_gmm_contour(gca,Priors,Mu,Sigma,[0 0 1]);


plot(XData(:,1),XData(:,2),'+r');
axis equal;
subplot(1,2,2);
hold on; grid on;
plot_gmm_contour(gca,Priors_j,Mu_j,Sigma_j,[0 0 1]);
plot(XData(:,1),XData(:,2),'+r');
axis equal;

