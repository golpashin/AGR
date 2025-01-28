% =========================================================================
% Project Name: Adaptive Grid Refinement for Optimal Feedback Control
% File Name: main.m and dependencies (See below)
% Author: Alen Golpashin
% Email: agolpa2@illinois.edu
% Copyright (c) 2024 Alen Golpashin
% Licensed under the MIT License. See LICENSE.txt for details.
% =========================================================================
%
% This project implements a reinforcement learning (RL) framework for 
% solving the Hamilton-Jacobi-Bellman (HJB) Partial Differential Equation
% using an adaptive computational grid. A Proximal Policy Optimization 
% (PPO) agent learns to adjust the grid by adding or removing (Quasi-Monte 
% Carlo sampled) nodes to reduce solution error, enhancing computational 
% efficiency and accuracy.
% Written on: MATLAB R2023b
%
% - main.m: Main script that configures and trains the PPO agent, 
%   setting up the environment, actor and critic networks, and training 
%   parameters.
%
% Function Dependency Summary:
%
% - add_nodes.m: Adds a specified percentage of nodes within a given 
%   radius around a sampled point, using Quasi-Monte Carlo sampling.
%
% - apply_action.m: Applies the action (node addition or removal) selected 
%   by the agent and updates the environment's state, including the grid 
%   and solution functions.
%
% - compute_error.m: Computes local error at a specified point to assess 
%   the accuracy of the control solution.
%
% - compute_reward.m: Calculates the reward for each action based on error 
%   reduction, node adjustment, and other criteria, guiding the agent 
%   towards learning the optimal policy.
%
% - compute_solution.m: Iterates the HJB solver over the grid to obtain 
%   the value function and control based on the current grid configuration.
%   The obtained solution is used as training data.
%
% - distance_ratio.m: Computes the ratio of the minimum normalized
%   distance to the boundary over the maximum normalized distance to the
%   boundary from a sampled point. 
%
% - domain_radius.m: Computes the radius of an xdims-dimensional ball that
%   has the same volume as the defined PDE domain.
%
% - hjb_grid.m: Defines the custom environment class, including states, 
%   actions, dynamics, and parameters for the adaptive grid-based 
%   control problem.
%
% - initial_grid.m: Initializes the grid with a specified number of nodes 
%   within a bounded domain. Quasi-Monte Carlo sampling is used to generate
%   the grid.
%
% - knn_density.m: Computes the k-nearest neighbor (k-NN) density estimate
%   for a given point.
%
% - main_cont.m: Sets up the training to continue or simulates an episode
%   using a trained agent.
%
% - precomputed_solution.mat: The precomputed solution data, containing
%   the reference solution, control, grid, and interpolators.
%   
% - prepare_action.m: Prepares the action to be applied to the environment.
%   It also clamps the action values to ensure they stay within defined
%   bounds, so that the training session does not crash.
%   
% - remove_nodes.m: Removes a percentage of nodes within a specified radius 
%   around a reference node, based on the agent's action.
%   
% - residual_integral.m: Calculates the integrated residual error over the 
%   entire grid, serving as a performance metric for the solution.
%
% - sample_points.m: Samples a random point from the domain. Local values
%   such as density, error, and distance are calculated at this point.
%
% - update_contour.m: Plots and updates the grid with options to overlay 
%   nodes on an error contour.
%
% - update_plot.m: Updates visualization plots during training to track 
%   the grid, error metrics, and agent performance.
%
% - update_surface.m: Updates a 3D surface plot representing the value 
%   function over the grid, allowing visual inspection of the
%   solution quality.

close all; clear; clc; % clean start

%% Set up parallel pool
CPU_cores = feature('numCores'); % get the number of available CPUs
fprintf('Number of available CPU cores: %d\n', CPU_cores); % notify user
poolobj = gcp('nocreate'); % check if a parallel pool exists
if isempty(poolobj)
    parpool('local', CPU_cores);
end

%% Set up the Environment
env = hjb_grid;

% Get observation and action specifications from the environment
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Create the Actor and Critic Networks
numObs = obsInfo.Dimension(1);
numAct = actInfo.Dimension(1);

% Actor Network
% Get action bounds
minR = actInfo.LowerLimit(1);
maxR = actInfo.UpperLimit(1);
minP = actInfo.LowerLimit(2);
maxP = actInfo.UpperLimit(2);

% Define common input path layer
commonPath = [
    featureInputLayer(numObs, 'Normalization', 'none', 'Name', 'comPathIn')
    fullyConnectedLayer(64, 'Name', 'commonFC1')
    layerNormalizationLayer("Name","commonLN1")
    reluLayer('Name', 'commonRelu1')
    fullyConnectedLayer(64, 'Name', 'commonFC2')
    layerNormalizationLayer("Name","commonLN2")
    reluLayer('Name', 'commonRelu2')
    fullyConnectedLayer(32, 'Name', 'comPathOut')
    ];

% Define mean value path
meanPath = [
    fullyConnectedLayer(32, 'Name', 'meanPathFC1')
    layerNormalizationLayer("Name","meanLN1")
    reluLayer('Name', 'meanRelu1')
    fullyConnectedLayer(numAct, 'Name', 'meanPathFC2')
    layerNormalizationLayer("Name","meanLN2")
    tanhLayer('Name', 'meanTanh')  
    ];

% functionLayers extracting each dimension separately
selectR = functionLayer(@(X) X(1,:), 'Name', 'selectR', 'Formattable', true);
selectP = functionLayer(@(X) X(2,:), 'Name', 'selectP', 'Formattable', true);

% Scaling for R dimension: map [-1,1] to [minR,maxR]
RScaling = scalingLayer('Name','RScaling', ...
    'Scale', (maxR - minR)/2, ...
    'Bias', (maxR + minR)/2);

% Scaling for PercentChange dimension: map [-1,1] to [minP,maxP]
PScaling = scalingLayer('Name','PScaling', ...
    'Scale', (maxP - minP)/2, ...
    'Bias', (maxP + minP)/2);

% Concatenate the scaled outputs back
concatActions = concatenationLayer(1, 2, 'Name','concatActions');

% Define standard deviation path without scaling
sdevPath = [
    fullyConnectedLayer(32, 'Name', 'stdPathFC1')
    layerNormalizationLayer("Name","stdLN1")
    reluLayer('Name', 'stdRelu1')
    fullyConnectedLayer(numAct, 'Name', 'stdPathFC2')
    layerNormalizationLayer("Name","stdLN2")
    softplusLayer('Name', 'Splus') % no scaling layer after this
    ];

% Assemble the layer graph for the actor
actorNet = layerGraph(commonPath);
actorNet = addLayers(actorNet, meanPath);
actorNet = addLayers(actorNet, sdevPath);

% Add the dimension selection and scaling layers
actorNet = addLayers(actorNet, selectR);
actorNet = addLayers(actorNet, RScaling);
actorNet = addLayers(actorNet, selectP);
actorNet = addLayers(actorNet, PScaling);
actorNet = addLayers(actorNet, concatActions);

% Connect paths
actorNet = connectLayers(actorNet, 'comPathOut', 'meanPathFC1');
actorNet = connectLayers(actorNet, 'comPathOut', 'stdPathFC1');

% Connect meanTanh output to selection layers
actorNet = connectLayers(actorNet, 'meanTanh', 'selectR');
actorNet = connectLayers(actorNet, 'meanTanh', 'selectP');

% Connect selection outputs to respective scaling layers
actorNet = connectLayers(actorNet, 'selectR', 'RScaling');
actorNet = connectLayers(actorNet, 'selectP', 'PScaling');

% Connect scaled outputs to concatenation
actorNet = connectLayers(actorNet, 'RScaling', 'concatActions/in1');
actorNet = connectLayers(actorNet, 'PScaling', 'concatActions/in2');

% Convert the actor network to dlnetwork
actorNet = dlnetwork(actorNet);

% Create the actor representation
actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    'ObservationInputNames', 'comPathIn', ...
    'ActionMeanOutputNames', 'concatActions', ...
    'ActionStandardDeviationOutputNames', 'Splus');

% Critic Network
criticLayerSizes = [64, 64];

criticNetwork = [
    featureInputLayer(numObs, 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(criticLayerSizes(1), 'Name', 'criticFC1')
    layerNormalizationLayer
    reluLayer('Name', 'criticRelu1')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'criticFC2')
    layerNormalizationLayer
    reluLayer('Name', 'criticRelu2')
    fullyConnectedLayer(1, 'Name', 'criticOutput')
    ];

% Convert critic network to dlnetwork
criticNetwork = dlnetwork(criticNetwork);

% Create the critic representation without 'OptimizerOptions'
critic = rlValueFunction(criticNetwork, obsInfo, 'ObservationInputNames', 'state');

%% Configure the PPO Agent Options
% Specify optimizer options
actorOpts = rlOptimizerOptions('LearnRate', 2e-5, 'GradientThreshold', 1);
criticOpts = rlOptimizerOptions('LearnRate', 1e-3, 'GradientThreshold', 1);

%% Set up the PPO agent
agentOpts = rlPPOAgentOptions(... % PPO agent settings
    'ExperienceHorizon', 1024, ... % Steps in each trajectory before learning; higher improves stability, lower boosts exploration.
    'MiniBatchSize', 512, ...     % Experiences per mini-batch; higher smooths updates, lower speeds training.
    'ClipFactor', 0.2, ...        % Clipping factor for PPO objective; higher risks instability, lower slows learning.
    'EntropyLossWeight', 0.05, ...% Weight for exploration; higher encourages exploration, lower favors determinism.
    'NumEpoch', 3, ...            % Optimization passes per mini-batch; higher deepens learning, lower requires more updates.
    'AdvantageEstimateMethod', 'gae', ... % Method for advantage estimation; 'gae' offers bias-variance trade-off.
    'GAEFactor', 0.95, ...        % Discount factor for GAE; higher favors long-term, lower favors immediate rewards.
    'DiscountFactor', 0.99, ...   % Discount factor for future rewards; higher emphasizes long-term gains.
    'ActorOptimizerOptions', actorOpts, ...
    'CriticOptimizerOptions', criticOpts); 

% Save optimizer states
agentOpts.InfoToSave.Optimizer = true;

% Create the PPO agent
agent = rlPPOAgent(actor, critic, agentOpts);

%% Set up training
trainOpts = rlTrainingOptions(... % training options
    'MaxEpisodes', 200000, ...
    'MaxStepsPerEpisode', env.param.max_steps, ...
    'ScoreAveragingWindowLength', 100, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'SaveAgentCriteria', 'EpisodeFrequency', ...
    'SaveAgentValue', 1000);

% Parallel Training Toggle
trainOpts.UseParallel = true;
trainOpts.ParallelizationOptions.Mode = 'async';

% Create and Configure a Data Logger
log_data = true; % true=log the specified data
log_to_file = false; % true=save logs to file
if log_data 
    if  ~log_to_file
        monitor = trainingProgressMonitor();
        logger = rlDataLogger(monitor);
    else
        logger = rlDataLogger();
        logDir = fullfile(pwd,"myDataLog");
        fileLogger.LoggingOptios.LoggingDirectory = logDir;
        fileLogger.LoggingOptions.FileNameRule = "episode<id>";
        logger.LoggingOptions.DataWriteFrequency = 10;
    end
    logger.AgentLearnFinishedFcn = @agentLearnFinishedFcn;
else
    logger = [];
end

%% Train the Agent
trainingStats = train(agent, env, trainOpts, Logger=logger);

%% Save Training Results
save('FinalAgent.mat', 'agent');
save('TrainingResults.mat', 'trainingStats');

%% Callback Function Definitions
function dataToLog = agentLearnFinishedFcn(data)
    dataToLog.ActorLoss = data.ActorLoss;
    dataToLog.CriticLoss = data.CriticLoss;   
end