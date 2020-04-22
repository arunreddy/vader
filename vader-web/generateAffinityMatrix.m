
function [A] = generateAffinityMatrix(X)

[m, n] = size(X)

[Edgeidx, Dist] = knnsearch(X, X, 'K', 11, 'IncludeTies', true);

fcn = @removeFirst; % remove the NN of node itself
Edgeidx =  cellfun(fcn, Edgeidx, 'UniformOutput', false);
Dist =  cellfun(fcn, Dist, 'UniformOutput', false);

sigma = std(X(:,:)).^2;
sigmamean = mean(sqrt(sigma));
sigmamean
% sigmamean = 0.05;
% fcn2 = @(x) exp(-1*x.^2); % from ICML 04 Harmonic paper
fcn2 = @(x) exp(-1/sigmamean*x); % from ICML 04 Harmonic paper

Dist = cellfun(fcn2, Dist, 'UniformOutput', false);
[NodesS, NodesT, EdgeWeights] = generateEdgeTable(Edgeidx, Dist);

A = full(sparse(NodesS, NodesT, EdgeWeights, m, m));
A = A + A';

end