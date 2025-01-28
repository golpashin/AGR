function [x_added] = add_nodes(point, x, R, P, param)
    %% Summary:
    % Adds a specified percentage of nodes within a radius R around a sampled point.
    % The number of nodes added is determined by P (percent_change) relative to the
    % total number of nodes within the radius R.
    %% Input:
    % point: coordinates of the sampled point
    % x: coordinates of the current nodes
    % R: radius of the region to add nodes in
    % P: percentage of nodes to add within the area defined by R
    % param: structure containing various constant parameters
    %% Output:
    % x_added: the coordinates of the newly added nodes
    
    
    % Initialize values
    x_added = [];
    x_added_valid = [];
    valid_count = 0; % number of valid nodes generated
    sequence_index = 1; % index for the generated sequence
    
    % Calculate the number of nodes to add (dN) based on P
    num_existing_nodes_within_R = sum(sqrt(sum((x - point).^2, 1)) <= R);
    dN = round(abs(P) * num_existing_nodes_within_R);
    
    % Return if no nodes need to be added
    if dN <= 0
        return;
    end

    % Create Sobol sequence generator for Quasi-Monte Carlo sampling
    sobolGen = sobolset(param.xdims, 'Skip', param.skip_QMC); % Skip initial points
    rng('shuffle'); % reset the random number generator
    sobolGen = scramble(sobolGen, 'MatousekAffineOwen'); % scramble for better distribution

    % Generate nodes up to dN within radius R, respecting bounds and duplicate tolerance
    while valid_count < dN
        % Generate one QMC point at a time
        QMCpoint = net(sobolGen, sequence_index);
        
        % Map the point to the circular domain
        theta = 2 * pi * QMCpoint(sequence_index,1); % Angle
        r = R * sqrt(QMCpoint(sequence_index,2)); % Radius (sqrt to maintain uniform distribution in circular area)
        
        % Convert to Cartesian coordinates relative to the sampled node
        x1 = point(1) + r * cos(theta);
        x2 = point(2) + r * sin(theta);
        
        % Check if point is within domain bounds
        if x1 >= param.bd(1, 1) && x1 <= param.bd(1, 2) && ...
           x2 >= param.bd(2, 1) && x2 <= param.bd(2, 2)
            
            % Check for duplicates with existing nodes
            overlap = false;
            for j = 1:size(x, 2)
                if norm([x1 - x(1, j), x2 - x(2, j)]) < param.duplicate_tol
                    overlap = true;
                    fprintf('WARNING: Duplicate node generated. Node is not added.\n');
                    break;
                end
            end
            
            % If no overlap, add to the valid points
            if ~overlap
                x_added_valid = [x_added_valid, [x1; x2]];
                valid_count = valid_count + 1;
            end
        end
        
        % Increment the index for the Sobol sequence generator
        sequence_index = sequence_index + 1;
    end

    % Return the valid added nodes
    x_added = x_added_valid;

    % % Plot for diagnostic testing
    % figure;
    % hold on;
    % plot(x_added(1,:), x_added(2,:), 'o');
    % theta_circle = linspace(0, 2*pi, 100);
    % x_circle = point(1) + R * cos(theta_circle);
    % y_circle = point(2) + R * sin(theta_circle);
    % plot(x_circle, y_circle, 'r-', 'LineWidth', 2);
    % rectangle('Position', [param.bd(1,1), param.bd(2,1), ...
    %     param.bd(1,2)-param.bd(1,1), param.bd(2,2)-param.bd(2,1)], ...
    %     'EdgeColor', 'b', 'LineWidth', 2);
    % hold off;
    % title('Generated QMC Grid');
    % xlabel('x_1');
    % ylabel('x_2');
    % grid on;
    % axis equal;
end

