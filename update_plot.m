function plot_grid = update_plot(x, error_norm, I, Iu, IsDone, plot_grid, param)
    %% Summary: 
    % Plots and updates the grid, including the optimal trajectory if IsDone is true.
    %% Input:
    % x: coordinates of grid nodes (2 x n matrix)
    % error_norm: current integrated error norm over the domain
    % I: Interpolation structure for value function
    % Iu: Interpolation structure for control policy
    % IsDone: Episode termination flag
    % plot_grid: grid plot handle
    % param: structure containing parameters and options
    %% Output:
    % plot_grid: Updated plot handle
    
    
    if param.plot_grid
        % Initialize or update grid plot
        if isempty(plot_grid) || ~isvalid(plot_grid)
            plot_grid = figure('Name', 'Dynamic Grid Update');
        else
            figure(plot_grid);
        end
        
        % Plot the current grid
        scatter(x(1, :), x(2, :), 5, 'filled');
        xlabel('x_1');
        ylabel('x_2');
        title(['Dynamic Grid Update | Integrated Error = ', num2str(error_norm)]);
        hold on;

        % Plot optimal trajectory if the episode is done
        if IsDone & param.plot_traj
            % Use the last entry of x as the initial state
            y0 = x(:, end);

            % Compute optimal time-to-go from the initial state
            T_opt = real(-log(1 - I(y0(1), y0(2)))); 

            % Define the dynamics
            odes = @(t, y) [
                (y(2) <= 1) * cos(Iu(y(1), y(2))) + ...
                (y(2) > 1) * ((y(2) - 1)^2 + 1) * cos(Iu(y(1), y(2))); % dx/dt
                
                (y(2) <= 1) * sin(Iu(y(1), y(2))) + ...
                (y(2) > 1) * ((y(2) - 1)^2 + 1) * sin(Iu(y(1), y(2)))  % dy/dt
            ];

            % Integrate the optimal trajectory
            [~, y] = ode15s(odes, [0, T_opt], y0);

            % Plot the trajectory
            plot(y(:, 1), y(:, 2), 'r-', 'LineWidth', 2);
            scatter(y0(1), y0(2), 100, 'g', 'filled'); % Initial state
            scatter(param.Target(1), param.Target(2), 100, 'b', 'filled'); % Target
            legend('Grid Nodes', 'Optimal Trajectory', 'Initial State', 'Target');
        end
        
        hold off;
        drawnow();
    end
end
