%% QEM with 2D world
clear all;
%% Load Value function
name = 'llr';
path_to_save = '/home/guillaume/MatlabWorkSpace/RL/2D_example/v-function_saved';
load([path_to_save '/' name '.mat']);
%% Load mex Value function
name = 'lwrcpp';
path_to_load = '/home/guillaume/MatlabWorkSpace/RL/2D_example/v-function_saved';
load([path_to_load '/' name]);

Q =  LWR_mex(q_mex_save.options);
Q.train(q_mex_save.X,q_mex_save.y);
%% Load data
Data  = loadAllMatFiles('/home/guillaume/MatlabWorkSpace/RL/2D_example/Data/record',true);
xurxp = create_tuples(Data);
xuv   = get_advantage_data(xurxp,Q,0.95);

%% plot advantage

Av = xuv(end,:);
Av(find(Av <= 0))=0;

close all;
figure; hold on;
plot(sort(Av));

%% Plot

Av = rescale(Av,min(Av),max(Av),0,1);

close all;
figure; 
subplot(1,2,1);
scatter(xuv(1,:),xuv(2,:),10,Av);
colorbar;

xs    = linspace(-10,10,100);
[X,Y] = meshgrid(xs,xs);
Z     = Q.f([X(:),Y(:)]);

subplot(1,2,2);
hold on;
surfc(X,Y,reshape(Z,size(X)));
colorbar; shading interp;

%% Learn initial policy

options = statset('Display','final','MaxIter',400);
GMModel = fitgmdist(xuv(1:end-1,:)',10,'Options',options,'CovarianceType','full','RegularizationValue',1e-05,'Replicates',5,'Start','plus');


Priors          = GMModel.ComponentProportion;
Mu              = GMModel.mu';
Sigma           = GMModel.Sigma;
in              = [1 2];
out             = [3 4];

%% Initialise policy

[Priors, Mu, Sigma] = EM_init_kmeans_cov(xuv(1:end-1,:), 10,'isotropic');

%% Run QEM

[Priors, Mu, Sigma, Pix] = QEM(xuv(1:end-1,:),Av',Priors, Mu, Sigma ,'full');


%% Plot GMR Policy
in              = [1 2];
out             = [3 4];

xs    = linspace(-10,10,25);
[X,Y] = meshgrid(xs,xs);

tXTest = [X(:),Y(:)];

[y, Sigma_y] = GMR(Priors, Mu, Sigma, tXTest', in, out);

yn = normr(y')';

close all;
figure; hold on; box on; grid on;
quiver(tXTest(:,1),tXTest(:,2),yn(1,:)',yn(2,:)',0.6);
