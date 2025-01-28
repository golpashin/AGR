function nodes_surface = update_contour(x, I, Iu, error_norm, nodes_surface, param)
    %% Summary: 
    % Plots and updates the grid with a contour of the error
    % computed over a sampled QMC grid.
    %% Input:
    % x: coordinates of grid nodes (2 x n matrix)
    % I: interpolation structure for the value function
    % Iu: interpolation structure for the control
    % error_norm: current integrated error norm over the domain
    % nodes_surface: grid plot handle
    % param: structure containing parameters and options
    %% Output:
    % nodes_surface: grid plot handle

    
    if param.nodes_surface

        %% Generate QMC Grid
        % Set up the Sobol sequence generator
        sobol_gen = sobolset(param.xdims, 'Skip', param.skip_QMC, 'Leap', param.leap_QMC);
        sobol_gen = scramble(sobol_gen, 'MatousekAffineOwen'); % Scramble for better distribution

        % Generate quasi-random samples within the domain bounds
        samples = net(sobol_gen, param.num_samples)'; % Transpose to match dimensions
        for i = 1:size(param.bd, 1)
            % Scale samples to the specified domain bounds
            samples(i, :) = param.bd(i, 1) + samples(i, :) * (param.bd(i, 2) - param.bd(i, 1));
        end

        %% Compute Error at Sampled Points
        error_values = zeros(1, param.num_samples);

        parfor i = 1:param.num_samples
            % Get sample coordinates
            x1 = samples(1, i);
            x2 = samples(2, i);

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

            % Compute error residual
            residual = abs(fGradV + 1); % Absolute value for error

            % Store the residual
            error_values(i) = residual;
        end

        %% Interpolation for Contour
        % Define a regular grid for contouring
        contour_samples_x = linspace(param.bd(1, 1), param.bd(1, 2), 1000);
        contour_samples_y = linspace(param.bd(2, 1), param.bd(2, 2), 1000);
        [X, Y] = meshgrid(contour_samples_x, contour_samples_y);
        Z = griddata(samples(1, :), samples(2, :), error_values, X, Y, 'cubic');

        %% Plot the Contour
        % Update or initialize the plot
        if isempty(nodes_surface) || ~isvalid(nodes_surface)
            % Initialize the plot
            nodes_surface = figure('Name', 'Dynamic Grid with Error Contour');
        else
            figure(nodes_surface);
        end

        % Plot the contour
        contourf(X, Y, Z, 20, 'LineColor', 'none');
        colormap('jet');
        colorbar;

        hold on;

        % Overlay the grid nodes
        scatter(x(1, :), x(2, :), 15, 'k', 'filled');

        % Add labels and title
        xlabel('x_1');
        ylabel('x_2');
        title(['Dynamic Grid Update | Integrated Error = ', num2str(error_norm)]);

        hold off;
        drawnow();
    end
end

%% Reverse Kruzkov transformation
function untransformed = R(transformed)
    % Reverse transformation for the value function
    untransformed = real(-log(1 - transformed));
end