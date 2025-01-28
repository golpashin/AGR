function [x, n, V, u, I, Iu] = apply_action(R, P, point, dist, x, n, V, u, I, Iu, param)
    %% Summary: 
    % Modifies the grid by adding or removing a percentage of nodes within a radius R
    % and interpolates the HJB PDE solution and control for the modified grid.
    %% Input:
    % R: Radius of the ball (xdims-ball) within which nodes are added or removed
    % P: Percentage of node population (within xdims-ball of radius R) to be added or removed
    % point: coordinates of the sampled point for the action
    % dist: distances from each node to the sampled point
    % x: current node coordinates
    % n: current number of nodes
    % V: current value function estimates
    % u: current control estimates
    % I: interpolator for value function
    % Iu: interpolator for control
    % param: structure with parameters and options
    %% Output:
    % x: updated node coordinates 
    % n: updated node count
    % V: updated value function
    % u: updated control
    % I: updated interpolators for value function
    % Iu: updated interpolator for control
    
    
    %% Initialize variables
    x_added = [];
    keep_idx = [];
    
    %% Grid update
    if P < 0
        
        % Remove nodes
        [x, keep_idx] = remove_nodes(x, n, P, dist, R);
        
    elseif P > 0 && R > 0
        
        % Add nodes
        [x_added] = add_nodes(point, x, R, P, param);
        
    else
        
        % Do nothing (grid x stays the same)
        
    end
    

    %% Update solution for new grid
    if ~isempty(x_added) || size(x,2) < n
    
        % If nodes were added
        if ~isempty(x_added)
            
            % Interpolate V and u at new nodes using interpolation
            V_added = I(x_added(1, :)', x_added(2, :)')';
            u_added = Iu(x_added(1, :)', x_added(2, :)')';
            % Add generated nodes to the beggining of the cooridnate matrix
            % (Note: this is important because it keeps the initial and
            % terminal nodes at the end of the array.)
            x = [x_added, x];
            V = [V_added, V];
            u = [u_added, u];
            n = size(x,2); % update node population

        % If nodes were removed
        elseif size(x,2) < n
        
            % Update V and u vectors after node removal 
            V_removal_pool = V(1:end-2);
            u_removal_pool = u(1:end-2);
            V_new = V_removal_pool(keep_idx);
            u_new = u_removal_pool(keep_idx);
            V = [V_new, V(:, end-1:end)];
            u = [u_new, u(:, end-1:end)];
            n = size(x,2); % update node population
        
        end
        
        % Recompute interpolators with updated grid using interpolation
        I = scatteredInterpolant(x(1, :)', x(2, :)', V', 'natural');
        Iu = scatteredInterpolant(x(1, :)', x(2, :)', u', 'natural');
    end
end
