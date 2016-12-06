function [Mu] = QGMC_EM(Data,Q,Priors, Mu0, Sigma,in,out)
%QGMC_EM Optimises the weighted Log-likelihood of Data
%
%   P(x,dx;\theta) = GMM(Data;Priors,Mu,Sigma)
%
%   P(dx|x;theta) =
%
% Inputs -----------------------------------------------------------------
%   o Data:    D x N array representing N datapoints of D dimensions.
%
%   o Q   :    N x 1  Q-values
%
%   o Priors0: 1 x K array representing the initial prior probabilities
%              of the K GMM components.
%   o Mu0:     D x K array representing the initial centers of the K GMM
%              components.
%   o Sigma0:  D x D x K array representing the initial covariance matrices
%              of the K GMM components.
%
%   o in :    (P x 1)
%
%
% Outputs ----------------------------------------------------------------
%
%   o Mu_out:  P x K array representing the centers of the K GMM components.
%
%% Criterion to stop the EM iterative update
loglik_threshold = 1e-10;


%% One set of GMM parameters for each data point N conditionals

K = length(Priors);
P = length(out);
N = size(Data,2);

Mu         = Mu0;
loglik_old = -realmax;
nbStep     = 0;

while 1
    
    % o h : N x K
    [h,Mu_c,Sigma_c] = GMC(Priors, Mu, Sigma, Data(in,:), in, out);
    %
    % o y  : (N x 1), likelihood of each point P(dx|x)
    % o ys : (N x K), likelihood of each point for individual components of P
    [y,ys]           = gmc_pdf(Data(out,:),h,Mu_c,Sigma_c);
    
    
    
    % (N x K) : responsibility factor
    rq     = ((h .* ys) ./ repmat(y,1,K)) .* repmat(Q,1,K);
    % (1 x K)
    sum_rq = sum(rq,1);
    
    
    for k=1:K
        
        % (P x Q)
        Ak = Sigma(out,in) * inv(Sigma(in,in));
        %  (N x P) = (N x Q) * (Q x P)
        tmp  = (Data(in,:)  - repmat(Mu(in,k),1,N))' * Ak';
        %  (N x P)
        %  (P x N) - (P x N)
       % tmp2 = (Data(out,:) - repmat(Mu(out,k),1,N))'; 
        
        % (P x P)
       % Sigma_dx_x_k = Sigma(out,out,k) - Sigma(out,in,k) * Sigma(in,in,k) * Sigma(in,out,k);
        
        
        %    (1 x P)
        %    (N x P)     (N x P)                    (N x P)  * (P x P)
        Mu(out,k) = sum(repmat(rq(:,k),1,P) .* ((Data(out,:)' + tmp)))  / sum_rq(k);
        
    end
    
    %% Stopping criterion %%%%%%%%%%%%%%%%%%%%
  
    [h,Mu_c,Sigma_c] = GMC(Priors, Mu, Sigma, Data(in,:), in, out);
    F                = gmc_pdf(Data(out,:),h,Mu_c,Sigma_c);  
    F(find(F<realmin)) = realmin;
    loglik = mean(log(F));
    
    %Stop the process depending on the increase of the log likelihood
    if abs((loglik/loglik_old)-1) < loglik_threshold
        break;
    end
    loglik_old = loglik;
    nbStep = nbStep+1;
    
end


end

