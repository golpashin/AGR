function [local_density, dist] = knn_density(point, x, n, param)
    %% Summary:
    % Computes the k-nearest neighbor (k-NN) density estimate for a given point.
    % The density is estimated using the distance to the k-th nearest neighbor
    % and the volume of a ball containing k neighbors.
    %% Input:
    % point: coordinates of the sampled point
    % x: coordinates of all nodes
    % n: total number of nodes
    %% Output:
    % local_density: estimated density at the sampled point
    % dist: distances from all nodes to the sampled point
    
    
    % Number of dimensions
    d = param.xdims;

    % Number of neighbors to use for the density estimation
    k = round(sqrt(n)); % rule of thumb: k ≈ sqrt(n)

    % Compute distances from the sampled point to all the nodes
    dist = sqrt(sum((x - point).^2, 1));

    % Sort distances and find the k-th nearest neighbor distance
    dist_sorted = sort(dist);
    R_k = dist_sorted(k);

    % Compute the volume of a d-dimensional ball with radius R_k
    V_d = (pi^(d / 2) / gamma(d / 2 + 1)) * R_k^d;

    % Estimate the density using the k-NN formula
    local_density = k / (n * V_d);
end
