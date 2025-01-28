function [R, P] = prepare_action(Action, domain_R, param)
    %% Summary:
    % Prepares the network outputs to be applied to the environment. Each 
    % action component is clipped to ensure it remains within the defined 
    % bounds. Without clipping, the training session may crash.   
    % For node removal, P (percent_change) is negative.
    % For node addition, P (percent_change) is positive.
    %% Input:
    % Action: Original action vector produced by the policy network
    % domain_R: radius of an xdims-dimensional ball with the same volume as the domain
    % param: Structure containing environment parameters, including action bounds
    %% Output:
    % R: Radius of the ball (xdims-ball) within which nodes are added or removed
    % P: Percentage of node population (within xdims-ball of radius R) to be added or removed
    
    
    %% Clip action values
    Action(1) = max(param.minR, min(param.maxR, Action(1))); % clip radius to stay within [minR, maxR]
    Action(2) = max(param.minP, min(param.maxP, Action(2))); % clip percent change to stay within [minP, maxP]
    
    %% Extract action values
    R = Action(1) * domain_R; % radius
    P = Action(2);            % percentage of nodes to add or remove
end
