function [hf,ph] = gmm_plot_modes(gmc,options,ph,hf_pos)
%GMM_PLOT_MODES 
%
%   input -----------------------------------------------------------------
%   
%       o gmc : struct, conditional Gaussian Mixture Model
%       
%       o hf  : figure handle
%
%
%% Options


X    = options.X;
Y    = options.Y;

Data = [X(:),Y(:)]';

% Data:  D x N 
Z = gmm_pdf(Data,gmc.Priors, gmc.Mu, gmc.Sigma);
Z = reshape(Z,size(X));



%% Plotting

if isempty(ph)
    
        % ------------ Plot Modes  ------------ %

        if isempty(hf_pos), hf(1) = figure; else hf(1) = figure('Position',hf_pos(1,:));end
        set(gcf,'color','w');
        [~,ph.contourf] = contourf(X,Y,Z);        
        title('Modes');
        axis equal;
else
        set(ph.contourf,'ZData',Z);
end

end

