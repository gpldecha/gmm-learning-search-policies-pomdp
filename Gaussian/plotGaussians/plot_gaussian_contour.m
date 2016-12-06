function handle = plot_gaussian_contour(haxes,Mu,Sigma,color,STDS,alpha )
%PLOT_GAUSSIAN_CONTOUR Summary of this function goes here
%   Detailed explanation goes here


if ~exist('STD','var'), STDS=1;end
if ~exist('color','var'), color=[0 0 1];end
if ~exist('alpha','var'), alpha=1;end

%for i=1:length(STDS)

for i=1:2
    handle(i) = plot_gaussian_ellipsoid(Mu,Sigma,i,100,haxes,alpha,color);
    set(handle,'LineWidth',2);
end
%end


end

