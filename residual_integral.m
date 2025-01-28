function [error_norm] = residual_integral(I, Iu, param)
    %% Summary:
    % Computes the L2-norm of the residual error of the HJB PDE over the domain
    % using Quasi-Monte Carlo sampling (Sobol sequence) to approximate the integral,
    % based on the dynamics.
    %% Input:
    % I: value function interpolator for the value function
    % Iu: control interpolator for optimal control
    % param: structure containing various constant parameters
    %% Output:
    % error_norm: L2-norm of the residual error over the domain
    
    
    % Set up the Sobol sequence generator
    sobol_gen = sobolset(param.xdims, 'Skip', param.skip_QMC + randi(param.num_samples), 'Leap', param.leap_QMC);
    rng('shuffle'); % reset the random number generator
    sobol_gen = scramble(sobol_gen, 'MatousekAffineOwen'); % Scramble for better distribution

    % Generate quasi-random samples within the domain bounds
    samples = net(sobol_gen, param.num_samples)'; % Transpose to match dimensions
    for i = 1:size(param.bd, 1)
        % Scale samples to the specified domain bounds
        samples(i, :) = param.bd(i, 1) + samples(i, :) * (param.bd(i, 2) - param.bd(i, 1));
    end

    % Define chunk size for hybrid processing
    chunk_size = param.error_chunk_size;
    num_chunks = ceil(param.num_samples / chunk_size);

    % Preallocate temporary storage for each chunk
    chunk_results = cell(1, num_chunks);

    % Parallelized processing
    parfor chunk = 1:num_chunks
        % Determine indices for this chunk
        start_idx = (chunk - 1) * chunk_size + 1;
        end_idx = min(chunk * chunk_size, param.num_samples);
        indices = start_idx:end_idx;

        % Extract samples for this chunk
        chunk_samples = samples(:, indices);

        % Preallocate residuals for this chunk
        chunk_residuals = zeros(1, length(indices));

        % Process each sample in this chunk
        for k = 1:length(indices)
            % Get sample coordinates
            x1 = chunk_samples(1, k);
            x2 = chunk_samples(2, k);

            % Interpolate control at the sample points
            u = Iu(x1, x2);

            %% Arc dynamics
            % Compute the speed c based on x2
            if x2 <= 1
                c_val = 1;
            else
                c_val = (x2 - 1)^2 + 1;
            end

            % Dynamics
            f1 = c_val * cos(u);
            f2 = c_val * sin(u);

            % Finite-difference directional gradient (central difference)
            forward_val = R(I(x1 + param.h * f1, x2 + param.h * f2));
            backward_val = R(I(x1 - param.h * f1, x2 - param.h * f2));
            fGradV = (forward_val - backward_val) / (2 * param.h);

            % Error residual based on the untransformed HJB PDE
            residual = fGradV + 1;

            % Check for NaN or Infinity in residual
            if isnan(residual)
                residual = 0; % Replace NaN with zero
                fprintf('WARNING: NaN residual value replaced by 0 at sample coordinates (x1, x2) = (%.4f, %.4f).\n', x1, x2);
            elseif isinf(residual)
                residual = 0; % Replace Infinity with zero
                fprintf('WARNING: Infinity residual value replaced by 0 at sample coordinates (x1, x2) = (%.4f, %.4f).\n', x1, x2);
            end

            % Square the residual and store it
            chunk_residuals(k) = residual^2;
        end

        % Store chunk results
        chunk_results{chunk} = chunk_residuals;
    end

    % Combine results from all chunks
    residuals_squared = horzcat(chunk_results{:});

    % Approximate the integral of residual^2 over the domain
    domain_area = prod(param.bd(:, 2) - param.bd(:, 1));  % Area (or volume) of the domain
    error_norm = sqrt((domain_area / param.num_samples) * sum(residuals_squared)); % L2-norm
end

%% Reverse Kruzkov transformation
function untransformed = R(transformed)
    % Reverse transformation for the value function
    untransformed = real(-log(1 - transformed));
end
