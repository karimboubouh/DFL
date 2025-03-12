function [A, B, C] = OPT(W)

% Suppress all warnings
warning('off', 'all');

%%% Updated Parameters for 50x50 Matrix

% c_k vector: Expanded to 50 elements by repeating the original 10-element pattern 5 times
v = repmat([1.5e4; 1e4; 1e4; 3e4; 2.5e4; 1.5e4; 1.5e4; 1e4; 1e4; 3e4], 5, 1);

% S vector: Expanded to 50 elements by repeating the original 10-element pattern 5 times
S = repmat([700; 30; 20; 80; 120; 600; 2000; 39; 459; 340], 5, 1);

kap = 1e-28;
I = 1;

% p vector: Expanded to 50 elements with the same pattern
p = 10 * repmat([0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1], 5, 1);

% band vector: Expanded to 50 elements with the same pattern
band = 2e7 * repmat([0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1], 5, 1);

dim = 500000;

% T_thres matrix: Expanded to 50x50 by tiling the original 10x10 matrix 5 times
T_thres_original = [
    1.2500   27.4152    1.0110    50.8624    1.0216    1.0967    1.0375    1.0221    1.0375    1.0375;
    1.0142    1.2500    1.0569    1.0833    1.0355    1.0569    1.0327    1.1577    1.0327    1.0569;
    1.0967    1.0569    1.2500    1.1577    1.1577    1.0789    1.0625    1.1577    1.1577    1.1577;
    1.0199    1.1577    1.0324    1.2500    1.1577    1.1577    1.1577    1.0891    1.0279    1.0967;
    1.0723    1.0569    1.0789   10.8181   10.2500    1.2500    1.1577    1.0327    1.1577    1.1577;
    1.0789    1.2500    1.1577   10.8181   10.1577    1.2500    1.0327    1.0723    1.0279    1.2500;
    1.0723    1.0318    1.1577    1.1577  117.4152    1.1577    1.2500    1.2500  117.4152    1.1577;
    1.0723    1.0789    1.1577    1.0789    1.1577    1.0789    1.0569    1.2500    1.1577    1.1577;
    1.0569    1.0327    1.1250    1.1577    1.1077    1.0789    1.0723    1.0327    1.2500    1.1077;
    1.0259    1.0789    1.1577    1.1577    1.1577    1.1577    1.0327    1.1577    1.1250    1.2500
];
T_thres = repmat(T_thres_original, 5, 5); % Now 50x50

% T matrix: Expanded to 50x50 by tiling the original 10x10 matrix 5 times
T_original = [
    0	30	0.00007	0.3	0.03	0.005	0.01	0.002	0.01	0.01;
    0.02	0	10	0.07	0.13	0.02	0.02	0.02	0.02	0.002;
    0.05	0.02	0	0.02	0.02	0.8	15	8	0.02	0.02;
    0.006	0.02	5	0	0.02	0.02	0.02	6	0.05	0.005;
    0.00001	0.02	0.008	0.1	0	15	0.02	0.00002	0.02	30;
    0.008	0.1	0.02	0.1	0.02	0	0.02	0.1	15	10;
    0.01	0.02	0.02	0.02	10	0.02	0	0	10	0.02;
    0.01	0.008	0.02	0.008	0.02	8	0.2	0	0.02	0.002;
    0	0.02	0.003	0.02	0.004	0.008	0.1	0.02	0	0.004;
    25	0.00008	200	0.02	0.02	0.02	0.02	0.02	0.003	0
];
T = repmat(T_original, 5, 5); % Now 50x50

C5 = 10;
R = 10;
a = 3;

zer = zeros(50, 50);
oness = ones(50, 50);
d = 0.5;

N = 50; % Updated from 10 to 50
X = 1;

% Define the optimization problem
prob = optimproblem;

% Creating the optimization variables
Z = optimvar('Z', N, N, 'LowerBound', zer, 'UpperBound', oness);
gam = optimvar('gam', 1, 'LowerBound', 0.1, 'UpperBound', 0.9);
f = optimvar('f', N, 'LowerBound', 1e6, 'UpperBound', 1e9);

% Define the constraints
consz = [];
for i = 1:N
    for j = 1:N
        if W(i,j) == 0
            consz = [consz, Z(i,j) == 0];
        end
    end
end

% Convergence constraint
consconv = norm(Z, 'fro')^2 <=  2 + a - 2*(1+a)/(gam*C5*sqrt(R));

% Constraints for rows and columns summing to 1
consw1 = [];
for i = 1:N
    consw1 = [consw1, sum(Z(i, :)) == 1];
end

consw2 = [];
for i = 1:N
    consw2 = [consw2, sum(Z(:, i)) == 1];
end

% Additional constraints based on temperature thresholds
alpha = 1e-2;
indic = alpha ./ (alpha + exp((-Z.^2) ./ alpha));
T_c = (I .* S .* v) ./ f;
temps = [];
% Using element-wise operations for large matrices
L = indic .* gam .* T;

for i = 1:N
    for j = 1:N
        temps = [temps, L(i,j) + T_c(j) <= T_thres(i,j)];
    end
end

% Assign constraints to the problem
prob.Constraints.consz = consz;
prob.Constraints.consconv = consconv;
prob.Constraints.consw1 = consw1;
prob.Constraints.consw2 = consw2;
prob.Constraints.temps = temps;

% Define the objective
E_c = sum(kap * I .* S .* v .* f.^2);

obj = sum(sum(indic .* (p .* T))) + 0.5 * norm(W - Z, 'fro');
% Alternatively, if you prefer the commented objective:
% obj = E_c + sum(sum(Z .* (T .* repmat(p, 1, N)))) + 0.5 * norm(W - Z, 'fro');

prob.Objective = obj;

% Initial guess
x0.Z = W;
x0.f = ones(N,1) * 1e6;
x0.gam = 0.25;

% Optimization options with timeout functionality
timeout = 280;
startTime = tic;

function stop = maxTimeFcn(~, optimValues, ~)
    elapsedTime = toc(startTime);
    if elapsedTime > timeout
        fprintf('Timeout reached: Stopping optimization at iteration %d.\n', optimValues.iteration);
        stop = true;
    else
        stop = false;
    end
end

opts = optimoptions("fmincon", ...
    'Display', 'none', ...
    'MaxIterations', 10000, ...
    'MaxFunctionEvaluations', 500000, ...
    'OutputFcn', @maxTimeFcn, ...
    'StepTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-8, ...
    'Algorithm', 'interior-point', ...
    'ScaleProblem', true);

% Solve the optimization problem
[sol, fval, eflag, output] = solve(prob, x0, 'Options', opts);

% Display Results
fprintf('\n========================================\n');
fprintf('Optimization Results:\n');
fprintf('Objective value: %f\n', fval);
fprintf('Exit flag: %d\n', eflag);
fprintf('Number of iterations: %d\n', output.iterations);
fprintf('Number of function evaluations: %d\n', output.funcCount);
fprintf('First-order optimality: %e\n', output.firstorderopt);
fprintf('Execution time: %.2f seconds\n', toc(startTime));
fprintf('========================================\n');

% Assign outputs based on optimization results
if eflag == -1
    A = -999;
    B = -999;
    C = -999;
else
    sol.Z(sol.Z < 0.09) = 0;
    A = sol.Z;  % Optimized Z matrix
    B = sol.f;  % Optimized f vector
    C = sol.gam; % Optimized gamma
end

end
