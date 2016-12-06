function [ X,i ] = gmm_sample(nbSamples,Priors,Mu,Sigma )
%GMM_SAMPLE Summary of this function goes here
%   Detailed explanation goes here



D = size(Mu,1);
X = zeros(D,nbSamples);
    
i = discretesample(Priors', nbSamples);
    
if D == 1
    
    for v=1:nbSamples
        
        X(v) = normrnd(Mu(i(v)), sqrt(Sigma(i(v))) );    
        
    end  
else
    
    for v = 1:nbSamples
        j = i(v);
        X(:,v) = mvnrnd(Mu(:,j),Sigma(:,:,j));
    end
    
end


end

