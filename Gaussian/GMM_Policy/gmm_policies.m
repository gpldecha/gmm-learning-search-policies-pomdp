function [u,Priors_c,Mu_c,Sigma_c] = gmm_policies(F,in,out,gmm,type,utmp)
%GET_ACTION
%
%   input ----------------------------------------------------
%
%       o F:   (D x 1), current feature vector
%
%       o in:  (1 x P), dimensions to condition on P(out|in)
%
%       o out: (1 x Q); dimensions to output (predictor)
%
%       o gmm: structur, containing Priors, Mus and Sigmas
%
%   output ---------------------------------------------------
%
%       o u: (1 x 3), action to take
%
%

F = F(:);

[Priors_c,Mu_c,Sigma_c] = GMC(gmm.Priors, gmm.Mu, gmm.Sigma, F, in, out);
D                       = size(Mu_c,2);
K                       = length(Priors_c);
Mu_c                    = squeeze(Mu_c);
Mu_c                    = Mu_c ./ repmat(sqrt(sum(Mu_c.^2)),D,1);

utmp                    = utmp(:);
utmp                    = utmp ./norm(utmp+realmin);

if strcmp(type,'GMR') == true
    y  = GMR(gmm.Priors, gmm.Mu, gmm.Sigma, F, in, out);
    u  = y;
elseif strcmp(type,'GMA') == true
    
    % if First time step take the GMR actions
    if sum(utmp) == 0
        u = GMR(gmm.Priors, gmm.Mu, gmm.Sigma, F, in, out);
        %     elseif bResample == true
        %         disp('bResample');
        %          u  = gmm_sample(1,Priors_c,Mu_c,Sigma_c);
        
    else
        % ensure consistency with the previous action
        
        if sum(isnan(utmp)) ~= 0 || sum(isinf(utmp)) ~= 0  || length(utmp) ~= D
            u  = gmm_sample(1,Priors_c,Mu_c,Sigma_c);
        else
            
            if size(Mu_c,2) ~= K
                disp('error');
            end
            
            u = gmc_velocity(Priors_c,Mu_c',utmp);
            
        end
    end
    
    if sum(isnan(utmp)) ~= 0 || sum(isinf(utmp)) ~= 0
        u  = gmm_sample(1,Priors_c,Mu_c,Sigma_c);
    end
    
elseif strcmp(type,'MC') == true
    u  = gmm_sample(1,Priors_c,Mu_c,Sigma_c);
    
end
u = u(:);




%u  = sample_gmm(1,Priors_c,Mu_c,Sigma_c);
%u  = mean(u);



end


