function [dist_ratio] = distance_ratio(point, param)
    %% Summary:
    % Computes the ratio of the minimum normalized distance to the boundary
    % over the maximum normalized distance to the boundary from a given point.
    % If the shape of the domain is changed (e.g., going from a square to
    % a rectangle with a different aspect ratio), normalizing each distance
    % by the respective domain dimension ensures that distances along each
    % axis are interpreted on the same relative scale. 
    % Each distance is normalized by the domain's respective dimension:
    % e.g. normalized distance in x = distance_x / (xmax - xmin)
    %% Inputs:
    % point: coordinates of the sampled point
    % param: structure containing the domain parameters
    %% Output:
    % dist_ratio: (minimum normalized distance) / (maximum normalized distance)
    
    
    % Extract point coordinates
    x1 = point(1);
    x2 = point(2);

    % Extract boundary limits
    x1min = param.bd(1, 1);
    x1max = param.bd(1, 2);
    x2min = param.bd(2, 1);
    x2max = param.bd(2, 2);

    % Calculate raw distances to each boundary side
    dist_x1min = abs(x1 - x1min);
    dist_x1max = abs(x1max - x1);
    dist_x2min = abs(x2 - x2min);
    dist_x2max = abs(x2max - x2);

    % Compute domain widths
    domain_width_x1 = x1max - x1min;
    domain_width_x2 = x2max - x2min;

    % Normalize distances by domain dimensions    
    dist_x1min_norm = dist_x1min / domain_width_x1; % for x1-boundaries, divide by domain_width_x1
    dist_x1max_norm = dist_x1max / domain_width_x1;    
    dist_x2min_norm = dist_x2min / domain_width_x2; % for x2-boundaries, divide by domain_width_x2
    dist_x2max_norm = dist_x2max / domain_width_x2;
    
    % Compute the normalized minimum and maximum distances
    dist_min_norm = min([dist_x1min_norm, dist_x1max_norm, dist_x2min_norm, dist_x2max_norm]);
    dist_max_norm = max([dist_x1min_norm, dist_x1max_norm, dist_x2min_norm, dist_x2max_norm]);

    % Compute the ratio of normalized distances
    dist_ratio = dist_min_norm / dist_max_norm; % range: [0,1]
end