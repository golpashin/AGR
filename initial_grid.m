function [x, n] = initial_grid(param)
    %% Summary:
    % This function generates a QMC initial grid using Sobol sequence with skip and scramble options.
    % The user can specify the number of points, margins, and other parameters.
    %% Input:
    % param: structure containing various constant parameters
    %% Output:
    % x: coordinates of the current nodes (rows: dimensions, columns: node index)
    % n: number of nodes on grid x
    
    
    % Create a Sobol sequence generator with skip
    sobolGen = sobolset(param.xdims, 'Skip', param.skip_QMC); % skip the first 'skip_QMC' sequence values
    
    % Scramble the sequence
    rng('shuffle'); % reset the random number generator
    sobolGen = scramble(sobolGen, 'MatousekAffineOwen');

    % Generate Sobol sequence points
    qmc_points = net(sobolGen, param.num_points);

    % Scale the points to the desired range
    x1_min = param.bd(1,1) + param.marg;
    x1_max = param.bd(1,2) - param.marg;
    x2_min = param.bd(2,1) + param.marg;
    x2_max = param.bd(2,2) - param.marg;
    qmc_points(:,1) = x1_min + (x1_max - x1_min) * qmc_points(:,1);
    qmc_points(:,2) = x2_min + (x2_max - x2_min) * qmc_points(:,2);

    x = qmc_points;

    % Check if the target and initial states are in the grid, add if not
    tol = 1e-15;  % Tolerance for floating point comparison
    ic_in_x = any(all(abs(x - param.IC') < tol, 2));
    if ~ic_in_x
        x = [x; param.IC'];
    end
    idx_IC = find(all(abs(x - param.IC') < tol, 2)); % initial state node index
    if abs(x(end,:) - param.IC') > tol % check if IC is at the end of x
        x = [x(1:idx_IC-1,:); x(idx_IC+1:end,:); param.IC'];
    end
    target_in_x = any(all(abs(x - param.Target') < tol, 2));
    if ~target_in_x
        x = [x(1:end-1,:); param.Target'; x(end,:)];
    end
    idx_Target = find(all(abs(x - param.Target') < tol, 2)); % terminal state node index
    if abs(x(end-1,:) - param.Target') > tol % check if Target is at the end-1 of x
        x = [x(1:idx_Target-1,:); x(idx_Target+1:end-1,:); param.Target';x(end,:)];
    end

    n = length(x); % Number of nodes
    fprintf('Initial grid: %.0f nodes were generated.\n', n);
    x = x'; % size(x) = 2 dimensions by n nodes

    % Check if the nodes are unique
    x_unique = unique(x','rows')'; % remove duplicates
    if length(x) ~= length(x_unique)
         error('Non-unique nodes found!');
    end

    %% Plot for diagnostic testing
    % figure; 
    % scatter(x(1,:),x(2,:),'.');
    % xlabel('x_1');
    % ylabel('x_2');
    % title('Initial Grid');
end
