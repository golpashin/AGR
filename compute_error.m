function [errorV] = compute_error(point, I, Iu, param)
    %% Summary:
    % Computes the local HJB error at a point using the dynamics.
    %% Input:
    % point: coordinates of the point
    % I: value function interpolator
    % Iu: control interpolator
    % param: structure containing various constant parameters
    %% Output:
    % errorV: normalized error value at the point
    
    
    %% Dynamics computation
    % Interpolate the control
    u = Iu(point(1), point(2));

    % Compute the speed c based on point(2)
    if point(2) <= 1
        c_val = 1;
    else
        c_val = (point(2) - 1)^2 + 1;
    end
    
    % Dynamics for forward and backward directions
    f1_forward = c_val * cos(u);
    f2_forward = c_val * sin(u);
    f1_backward = -c_val * cos(u);
    f2_backward = -c_val * sin(u);
    
    %% Finite-difference directional gradient (central difference)
    forward_val = R(I(point(1) + param.h * f1_forward, point(2) + param.h * f2_forward));
    backward_val = R(I(point(1) + param.h * f1_backward, point(2) + param.h * f2_backward));
    fGradV = (forward_val - backward_val) / (2 * param.h);
    
    %% Error residual based on the untransformed HJB PDE
    errorV = abs(fGradV + 1); % absolute value of local error
    
    %% Check for NaN or Infinity in error
    if isnan(errorV)
        errorV = 0; % Replace NaN with zero
        fprintf('WARNING: NaN error value replaced by 0 at the specified node.\n');
    elseif isinf(errorV)
        errorV = 0; % Replace Infinity with zero
        fprintf('WARNING: Infinity error value replaced by 0 at the specified node.\n');
    end
    
    %% Reverse Kruzkov transformation
    function untransformed = R(transformed)
        % Reverse transformation for the value function
        untransformed = real(-log(1 - transformed));
    end
    
end
