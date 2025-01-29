function [V, u, I, Iu] = compute_solution(x, n, param)
    %% Summary: 
    % Computes the solution on the initial grid x. The solution guess 
    % is a naive guess and there is no need for a control guess. The
    % solution guess is then iterated to generate a solution value
    % function surface for training.
    %% Input:
    % x: coordinates of grid nodes (2 x n matrix)
    % n: number of nodes
    % param: structure containing various constant parameters
    %% Output:
    % V: updated value function after iterating the guess value function
    % u: updated control policy
    % I: interpolator for the value function
    % Iu: interpolator for the control
    
    
    %% Generate initial guesses for V and u
    V = param.initial_guess_V * ones(1, n);
    V(end-1) = 0; % terminal node
    % Notify user
    fprintf('Initial guess is obtained.\n');
    
    %% Control grid options
    du = param.du; % grid resolution
    lb = -pi; % control lower bound
    ub = pi; % control upper bound
    u_grid = linspace(lb, ub, du);
    cos_vals = cos(u_grid); % precompute cosines for all controls
    sin_vals = sin(u_grid); % precompute sines for all controls
    
    %% Time step assignment
    h = param.h;

    %% Set chunk size for hybrid approach
    chunk_size = param.chunk_size; % number of nodes per chunk
    num_chunks = ceil(n / chunk_size);

    %% Fixed-point iterations / Finding the solution at grid nodes 
    fprintf('Now iterating the initial guess...\n'); % notify user
    iter = 0; % initialize iteration counter
    check_error = 1; % initialize error flag
    while sum(check_error) > 0
        % Copy value function for convergence error computation
        U = V;

        % Kruzkov-transformed value function interpolation
        I = scatteredInterpolant(x(1,:)', x(2,:)', V', 'natural');
        
        % Preallocate temporary results for each chunk
        V_temp = cell(1, num_chunks);
        u_temp = cell(1, num_chunks);
        
        parfor chunk = 1:num_chunks
            % Determine the indices for this chunk
            start_idx = (chunk - 1) * chunk_size + 1;
            end_idx = min(chunk * chunk_size, n);
            indices = start_idx:end_idx;

            % Preallocate results for this chunk
            V_chunk = zeros(1, length(indices));
            u_chunk = zeros(1, length(indices));

            % Vectorized processing for nodes in this chunk
            x0_chunk = x(:, indices); % nodes in this chunk
            
            % Handle target node separately
            for k = 1:length(indices)
                idx = indices(k);
                x0 = x(:, idx); % individual node
                if norm(x0 - param.Target) < 1E-10
                    V_chunk(k) = 0;
                    u_chunk(k) = 0;
                    continue; % skip further computation for target node
                end

                % Precompute dynamics constant
                if x0(2) <= 1
                    c_val = 1;
                else
                    c_val = (x0(2) - 1)^2 + 1;
                end

                % Compute all candidate next states for u_grid
                x_next_all = x0 + h * c_val * [cos_vals; sin_vals];

                % Check domain validity
                valid_mask = all(x_next_all >= param.bd(:, 1) & x_next_all <= param.bd(:, 2), 1);

                % Initialize candidate values
                V_candidates = inf(1, length(u_grid)); % default to inf for invalid points
                if any(valid_mask)
                    valid_x_next = x_next_all(:, valid_mask); % filter valid next states
                    V_next = I(valid_x_next(1, :), valid_x_next(2, :)); % interpolate valid points
                    V_candidates(valid_mask) = exp(-h) * V_next + 1 - exp(-h);
                end

                % Find optimal control
                [V_min, idx_u] = min(V_candidates);
                u_optimal = u_grid(idx_u);

                % Update results for this node
                V_chunk(k) = V_min;
                u_chunk(k) = u_optimal;
            end

            % Assign chunk results to temporary storage
            V_temp{chunk} = V_chunk;
            u_temp{chunk} = u_chunk;
        end

        % Combine results from all chunks
        for chunk = 1:num_chunks
            start_idx = (chunk - 1) * chunk_size + 1;
            end_idx = min(chunk * chunk_size, n);
            indices = start_idx:end_idx;
            V(indices) = V_temp{chunk};
            u(indices) = u_temp{chunk};
        end
        
        % Check convergence
        fp_dV = abs(V - U);
        check_error = fp_dV(:) > param.tol_fp;
    
        % Notify user
        iter = iter + 1;
        if param.iter_info_all
            fprintf('iteration: %.0f\n', iter);
            fprintf('Current error inf-norm: %.15f\n', norm(fp_dV,'inf'));
            fprintf('Current error 2-norm: %.15f\n', norm(fp_dV));
            fprintf('\n');
        end
    end

    % Construct the control interpolation model for the current grid
    Iu = scatteredInterpolant(x(1, :)', x(2, :)', u', 'natural');
    % Save solution data to file
    save('precomputed_solution.mat', 'x', 'n', 'V', 'u', 'I', 'Iu');
    
    % Plot the solution
    if ~param.plot_surface
        figure;
        tri = delaunay(x(1, :), x(2, :)); % Triangulate the scattered data
        trisurf(tri, x(1, :), x(2, :), real((-log(1-V))')); % Plot the surface
        xlabel('x_1');
        ylabel('x_2');
        zlabel('V')
        title('Solution Surface');
        drawnow()
    end

    fprintf('Initial (iterated) solution guess is obtained.\n\n'); % inform user
end
