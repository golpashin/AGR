%% Summary:
% This code loads a pre-trained agent from the 'savedAgents' folder, and
% continues the training process. The training options can be modified in
% the 'TrainingOptions' struct. The code can also simulate the environment
% for a trained agent.

close all; clear; clc; % clean start

%% Set up parallel pool
CPU_cores = feature('numCores'); % get the number of available CPUs
fprintf('Number of available CPU cores: %d\n', CPU_cores); % notify user
poolobj = gcp('nocreate'); % check if a parallel pool exists
if isempty(poolobj)
    parpool('local', CPU_cores);
end

%% Train or Simulate
Training = false; % true=continue training instead of simulating 
Agent_Name = 'Agent104000.mat'; % name of the agent to be loaded

%% Set up the Environment
env = hjb_grid;

%% Agent
% Load previously trained agent to continue training
addpath('savedAgents');
load(Agent_Name ,'saved_agent');
load(Agent_Name ,'savedAgentResult');
agent = saved_agent;

% Modified training options
% savedAgentResult.TrainingOptions.UseParallel = false;
% savedAgentResult.TrainingOptions.ParallelizationOptions.Mode = 'async';

if Training
    % Continue training the agent
    disp('Continue training pre-trained model')
    trainingStats = train(agent, env, savedAgentResult);
    
    % Save Training Results
    save('FinalAgent.mat', 'agent');
    save('TrainingResults.mat', 'trainingStats');
else
    % Simulate trained agent 
    simOptions = rlSimulationOptions('MaxSteps', env.param.max_steps);
    sim(env, agent, simOptions);
end