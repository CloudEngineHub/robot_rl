close all;  clc;

% 1. Load the log as a table (keeps headers)
T = readtable('reward_terms.csv');   % first column must be 'step'
%%
% 2. Identify the reward columns automatically
varNames   = T.Properties.VariableNames;
rewardCols = setdiff(varNames, {'step'});   % all but the step counter

% 3. Generate a figure for every reward term
for k = 1:numel(rewardCols)
    figure('Name', rewardCols{k}, 'NumberTitle', 'off');
    plot(T.step, T.(rewardCols{k}), 'LineWidth', 1.2);
    grid on;
    xlabel('Simulation step');
    ylabel(rewardCols{k}, 'Interpreter', 'none');
    title(strrep(rewardCols{k}, '_', '\_'));  % preserve underscores
end
%%

dt   = 0.02;                % sec between physics steps (✓ adjust if needed)
time = T.step * dt;         % elapsed simulation time
currVel     = T.curr_vel;
cmdVelGiven = T.cmd_vel_given;

% 3. Low-pass filter on curr_vel  (Butterworth -> lowpass)
Fs   = 1/dt;                % sampling rate  (Hz)
Fc   = 0.1;                   % cut-off (Hz) – change to taste
currVel_lp = lowpass(currVel, Fc, Fs);

% 4. Plot all three curves
figure('Name','Velocity-related rewards','NumberTitle','off'); hold on;
plot(time, currVel,     '-',  'LineWidth',1.3);
plot(time, cmdVelGiven, '--', 'LineWidth',1.3);
plot(time, currVel_lp,  '-.', 'LineWidth',1.5);
grid on;

% 5. Decorations
xlabel('Time [s]');
ylabel('Reward value');
title('Current vs. Command Velocity Rewards (with LPF)');
legend({'curr\_vel','cmd\_vel\_given','curr\_vel LPF'}, ...
       'Interpreter','none','Location','best');