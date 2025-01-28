function [Reward, IsDone] = compute_reward(error_curr, error_init, n_curr, n_init, steps, Action, param)
    %% Summary:
    % Computes the reward for each step in the RL environment based on the
    % error reduction, node count changes.
    %% Inputs:
    % error_curr: current error at the current step  
    % error_init: initial error at the beginning of the episode
    % n_curr: current number of nodes
    % n_init: initial number of nodes at the beginning of the episode
    % steps: number of steps in the current episode 
    % Action: action for the current step (unprepared action)
    % param: structure containing the reward parameters
    %% Outputs:
    % Reward: Computed reward based on the error and node changes
    % IsDone: epsiode termination flag
    
    
    %% Persistent variables for cumulative rewards
    if param.terminal_reward
        persistent cumulative_R_E cumulative_R_N total_reward;
    end

    %% Definitions
    Rmin = -1; % lowest step reward per component
    
    %% Reward components
    % Error change (normalized to initial)
    delta_E = 1 - (error_curr / error_init);
    R_E = max(Rmin, delta_E);
    
    % Node count change (normalized to initial)
    delta_N = 1 - (n_curr / n_init);
    R_N = max(Rmin, delta_N);
    
    %% Action Penalization
    action_penalty_R = 0;
    action_penalty_P = 0;
    
    % Calculate action penalties based on distance from allowable ranges
    if param.action_penalty
        % Normalize each action separately (normalized to action range)
        if Action(1) < param.minR
            action_penalty_R = - abs((Action(1) - param.minR) / (param.maxR - param.minR));
        elseif Action(1) > param.maxR
            action_penalty_R = - abs((Action(1) - param.maxR) / (param.maxR - param.minR));
        end
        
        if Action(2) < param.minP
            action_penalty_P = - abs((Action(2) - param.minP) / (param.maxP - param.minP));
        elseif Action(2) > param.maxP
            action_penalty_P = - abs((Action(2) - param.maxP) / (param.maxP - param.minP));
        end
        
        % Penalty range: [Rmin, 0]
        action_penalty_R = max(Rmin, action_penalty_R);
        action_penalty_P = max(Rmin, action_penalty_P);
        normalized_penalty = action_penalty_R + action_penalty_P;
    else
        normalized_penalty = 0;
    end
    
    %% Define terminal rewards  
    % Episode termination conditions
    IsDone = false;   
    if n_curr < param.min_nodes || n_curr > param.max_nodes
        Reward = param.max_steps * Rmin; % assign penalty for exceeding node limits
        IsDone = true; % failure
    else
        Reward = R_E + R_N + normalized_penalty; % Summation of rewards
        % Check if step count exceeds maximum step count
        if steps >= param.max_steps
            IsDone = true;
        end
    end
    
    %% Display reward details if enabled
    if param.reward_info
        fprintf('====================================================================\n');
        fprintf('| Metric                        | Value         \n');
        fprintf('|-------------------------------|---------------\n');        
        fprintf('| Normalized Error Reward (R_E) | %.4f          \n', R_E);
        fprintf('| Normalized Node Reward (R_N)  | %.4f          \n', R_N);
        fprintf('| Total Reward                  | %.4f          \n', Reward);
        fprintf('| Action Penalty                | %.4f          \n', normalized_penalty);
        fprintf('| Current Error                 | %.4f          \n', error_curr);
        fprintf('| Reference Error               | %.4f          \n', error_init);
        fprintf('| Current Nodes                 | %d            \n', n_curr);
        fprintf('| Reference Nodes               | %d            \n', n_init);        
        fprintf('====================================================================\n\n');
    end
    
    % Report total reward at the end of the episode
    if param.terminal_reward
        % initialize sums
        if steps <= 1
            cumulative_R_E = 0;
            cumulative_R_N = 0;
            total_reward = 0;
        end        
        % Accumulate rewards
        cumulative_R_E = cumulative_R_E + R_E;
        cumulative_R_N = cumulative_R_N + R_N;
        total_reward = total_reward + Reward;
        % Report if episode is done
        if IsDone 
            fprintf('Episode Summary:\n');
            fprintf('====================================================================\n');
            fprintf('| Total R_E                     | %.4f          \n', cumulative_R_E);
            fprintf('| Total R_N                     | %.4f          \n', cumulative_R_N);
            fprintf('| Total Reward                  | %.4f          \n', total_reward);
            fprintf('====================================================================\n\n');
            % Clear persistent variables
            clearvars -global cumulative_R_E cumulative_R_N total_reward;
        end
    end
end