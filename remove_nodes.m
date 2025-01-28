function [x, keep_idx] = remove_nodes(x, n, P, dist, R)
    %% Summary: 
    % Removes a percentage of nodes randomly within a specified radius R
    % around a sampled point.
    %% Input:
    % x: coordinates of the nodes
    % n: number of nodes in the grid x
    % P: percentage of nodes to remove within radius R
    % dist: distances from all nodes to the sampled point
    % R: radius within which nodes may be removed
    %% Output:
    % x: new grid x after nodes are removed
    % keep_idx: indices of the remaining nodes in the original vector x,
    % to maintain correspondence of 'u' and 'V' with the modified x
    
    
    % Exclude initial and terminal nodes from removal consideration
    x_removal_pool = x(:, 1:end-2);
    dist_to_point = dist(1:end-2); % distances excluding the last two nodes

    % Identify nodes within the specified radius R
    nodes_within_R = find(dist_to_point <= R);

    % Calculate number of nodes to remove based on P
    num_nodes_within_R = length(nodes_within_R);
    num_to_remove = round(abs(P) * num_nodes_within_R);

    % If there are nodes to remove
    if num_to_remove > 0
        % Randomly select nodes to remove
        nodes_to_remove = randsample(nodes_within_R, min(num_to_remove, num_nodes_within_R));
        
        % Logical index array to keep nodes not in nodes_to_remove
        keep_idx = true(1, n-2);
        keep_idx(nodes_to_remove) = false;

        % Create new x by excluding removed nodes and keeping the last two nodes intact
        x_new = x_removal_pool(:, keep_idx);
        x = [x_new, x(:, end-1:end)]; % append the fixed nodes back
    else
        % If no nodes are removed, retain the original set with the fixed nodes
        x = [x_removal_pool, x(:, end-1:end)];
        keep_idx = true(1, n-2);
    end
end
