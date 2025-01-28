function [plot_surface] = update_surface(x, V, plot_surface, param)
    %% Summary: 
    % Plots and updates the value function surface.
    %% Input:
    % x: coordinates of grid nodes (2 x n matrix)
    % V: value function
    % plot_surface: value function plot handle
    % param: structure containing parameters and options
    %% Output:
    % plot_surface: value function handle
    
    
    if param.plot_surface

        if isempty(plot_surface) || ~isvalid(plot_surface)
            % Initialize grid plot
            plot_surface = figure('Name', 'Dynamic Solution Surface Update');
            tri = delaunay(x(1, :), x(2, :)); % Triangulate the scattered data
            trisurf(tri, x(1, :), x(2, :), real((-log(1-V))')); % Plot the surface
            xlabel('x_1');
            ylabel('x_2');
            zlabel('V')
            title('Dynamic Solution Surface Update');
            drawnow();
        else
            % Update the grid plot
            figure(plot_surface);
            tri = delaunay(x(1, :), x(2, :)); % Triangulate the scattered data
            trisurf(tri, x(1, :), x(2, :), real((-log(1-V))')); % Plot the surface
            xlabel('x_1');
            ylabel('x_2');
            zlabel('V')
            title('Dynamic Solution Surface Update');
            drawnow();
        end
    end
end

