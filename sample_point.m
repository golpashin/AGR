function [point] = sample_point(param)
    %% Summary:
    % Samples a point uniformly at random from the domain boundary.
    %% Input:
    % param: structure containing parameters, including domain boundary (param.bd)
    %% Output:
    % node: randomly sampled point within the domain
    
    
    % Preallocate the sampled point
    point = zeros(param.xdims, 1);
    
    % Sample each dimension independently within the domain bounds
    for i = 1:param.xdims
        point(i) = param.bd(i, 1) + (param.bd(i, 2) - param.bd(i, 1)) * rand();
    end
end