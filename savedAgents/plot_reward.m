clear all; close all; clc;

% Find the latest Agent file
files = dir('Agent*.mat');
if isempty(files)
    error('No agent files found in the current directory.');
end

% Extract agent numbers and find the maximum
agentNumbers = arrayfun(@(f) sscanf(f.name, 'Agent%d.mat'), files);
[maxAgentNum, idx] = max(agentNumbers);

% Load the corresponding file
load(files(idx).name, 'savedAgentResult');

% Extract the episode index and rewards
episodeIndex = savedAgentResult.EpisodeIndex;
episodeRewards = savedAgentResult.EpisodeReward;

% Smoothing window size
windowSize = 1000;

% Calculate rolling mean and rolling standard deviation
rollingMean = movmean(episodeRewards, windowSize);
rollingStd = movstd(episodeRewards, windowSize);

% Define the rolling variance band
upperBand = rollingMean + rollingStd; % +1 std deviation
lowerBand = rollingMean - rollingStd; % -1 std deviation

% Plot the rewards as individual points
figure;
scatter(episodeIndex, episodeRewards, 3, 'filled'); % Scatter plot for points
hold on;

% Plot the rolling mean
plot(episodeIndex, rollingMean, 'r-', 'LineWidth', 2); % Rolling mean line

% Fill the rolling variance band
fill([episodeIndex; flipud(episodeIndex)], ...
     [upperBand; flipud(lowerBand)], ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Variance band with transparency

% Add labels, title, and legend
xlabel('Episode Number');
ylabel('Episode Reward');
title(['Episode Rewards - Episode ', num2str(maxAgentNum)]);
legend({'Episode Rewards', 'Rolling Mean', '±1 Std Dev Band'}, 'Location', 'best');
grid on;
xlim([min(episodeIndex) max(episodeIndex)]); % tight x-axis
hold off;

% Save the figure as JPEG
saveas(gcf, ['Agent', num2str(maxAgentNum), '_EpisodeRewards.jpg']);
