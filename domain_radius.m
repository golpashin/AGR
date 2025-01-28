function [domain_R] = domain_radius(param)
    %% Summary:
    % Computes the radius of an xdims-dimensional ball that has the same
    % volume as the defined domain. Given a domain defined by bd, the
    % volume of the domain is:
    %   domain_V = Π_i (bd(i,2) - bd(i,1))
    % For an n-dimensional ball, the volume is given by:
    %   V_n(R) = (π^(n/2) / Γ((n/2) + 1)) * R^n
    % Setting V_n(R) = domain_V and solving for R:
    %   domain_R = R = [ (domain_V * Γ((n/2) + 1)) / π^(n/2) ]^(1/n)
    %% Inputs:
    % param: structure containing the domain parameters
    %% Outputs:   
    % domain_R: radius of an n-dimensional ball with the same volume as domain_V    
    
    
    % Compute the lengths of each dimension
    dims_length = zeros(param.xdims,1);
    for i = 1:param.xdims
        dims_length(i) = param.bd(i,2) - param.bd(i,1);
    end

    % Compute the domain volume
    domain_V = prod(dims_length);

    % Compute the radius R of the n-ball with the same volume as domain_V
    domain_R = ((domain_V * gamma(param.xdims/2 + 1)) / (pi^(param.xdims/2)))^(1/param.xdims);
end
