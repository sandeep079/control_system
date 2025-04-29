% Simplified Particle Filter in Octave
clear; close all;
graphics_toolkit("gnuplot"); % Ensure plotting works

% ===== Parameters =====
N = 500;               % Number of particles
T = 50;                % Time steps
process_noise = 0.3;   % Motion noise (std dev)
measurement_noise = 2; % Measurement noise (std dev)

% ===== Ground Truth & Measurements =====
x_true = cumsum(process_noise * randn(1, T)); % True position (random walk)
z = x_true + measurement_noise * randn(1, T); % Noisy measurements

% ===== Initialize Particles =====
particles = randn(1, N) * 5; % Random initial guesses
weights = ones(1, N) / N;    % Uniform weights
x_est = zeros(1, T);         % Estimated state

% ===== Particle Filter Loop =====
for t = 1:T
    % --- Prediction: Move particles randomly ---
    particles = particles + process_noise * randn(1, N);

    % --- Update: Weight particles based on measurement ---
    error = z(t) - particles;
    weights = exp(-0.5 * error.^2 / measurement_noise^2);
    weights = weights / sum(weights); % Normalize

    % --- Resampling (Stochastic Universal Sampling) ---
    new_indices = randsample(1:N, N, true, weights);
    particles = particles(new_indices);

    % --- Estimate: Mean of particles ---
    x_est(t) = mean(particles);
end

% ===== Plot Results =====
figure;
plot(1:T, x_true, 'b-', 'LineWidth', 2, 'DisplayName', 'True Position');
hold on;
plot(1:T, z, 'ro', 'MarkerSize', 4, 'DisplayName', 'Noisy Measurements');
plot(1:T, x_est, 'k--', 'LineWidth', 2, 'DisplayName', 'Particle Filter');
legend;
xlabel('Time Step');
ylabel('Position');
title('Simplified Particle Filter (1D Tracking)');
grid on; % Now works!
