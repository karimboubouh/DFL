function [A, B, C] = OPT_2(W, R)
    % Start a parallel pool using all available cores
    if isempty(gcp('nocreate'))
        parpool; % Use all available cores
    end

    % Suppress all warnings
    warning('off', 'all');

    %%% this c_k
    v = [1.5e4; 1e4; 1e4; 3e4; 2.5e4; 1.5e4; 1.5e4; 1e4; 1e4; 3e4];
    S = [700; 30; 20; 80; 120; 600; 2000; 39; 459; 340];
    kap = 1e-28;
    I = 1;
    p = 10 * [0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1];
    band = 2e7 * [0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1; 0.1];
    dim = 500000;

    T_thres = [1.2500 27.4152 1.0110 50.8624 1.0216 1.0967 1.0375 1.0221 1.0375 1.0375;
               1.0142 1.2500 1.0569 1.0833 1.0355 1.0569 1.0327 1.1577 1.0327 1.0569;
               1.0967 1.0569 1.2500 1.1577 1.1577 1.0789 1.0625 1.1577 1.1577 1.1577;
               1.0199 1.1577 1.0324 1.2500 1.1577 1.1577 1.1577 1.0891 1.0279 1.0967;
               1.0723 1.0569 1.0789 10.8181 10.2500 1.2500 1.1577 1.0327 1.1577 1.1577;
               1.0789 1.2500 1.1577 10.8181 10.1577 1.2500 1.0327 1.0723 1.0279 1.2500;
               1.0723 1.0318 1.1577 1.1577 117.4152 1.1577 1.2500 1.2500 117.4152 1.1577;
               1.0723 1.0789 1.1577 1.0789 1.1577 1.0789 1.0569 1.2500 1.1577 1.1577;
               1.0569 1.0327 1.1250 1.1577 1.1077 1.0789 1.0723 1.0327 1.2500 1.1077;
               1.0259 1.0789 1.1577 1.1577 1.1577 1.1577 1.0327 1.1577 1.1250 1.2500];

    T = [0      30      0.00007 0.3     0.03    0.005   0.01    0.002   0.01    0.01;
         0.02   0       10     0.07    0.13    0.02    0.02    0.02    0.02    0.002;
         0.05   0.02    0      0.02    0.02    0.8     15      8       0.02    0.02;
         0.006  0.02    5      0       0.02    0.02    0.02    6       0.05    0.005;
         0.00001 0.02   0.008  0.1     0       15      0.02    0.00002 0.02    30;
         0.008  0.1     0.02   0.1     0.02    0       0.02    0.1     15      10;
         0.01   0.02    0.02   0.02    10      0.02    0       0       10      0.02;
         0.01   0.008   0.02   0.008   0.02    8       0.2     0       0.02    0.002;
         0      0.02    0.003  0.02    0.004   0.008   0.1     0.02    0       0.004;
         25     0.00008 200    0.02    0.02    0.02    0.02    0.02   0.003   0];

    C5 = 1; % 2
    a = 4; % 3
    zer = zeros(10, 10);
    oness = ones(10, 10);
    d = 0.5;
    N = 10;
    X = 1;

    % Define optimization problem and variables
    prob = optimproblem;
    Z = optimvar('Z', N, N, 'LowerBound', zer, 'UpperBound', oness);
    gam = optimvar('gam', 1, 'LowerBound', 0.1, 'UpperBound', 0.9);
    f = optimvar('f', N, 'LowerBound', 1e8, 'UpperBound', 5e9);

    % --- Separate constraints on Z based on W ---
    % For indices where W(i,j)==0, we require Z(i,j)==0.
    consz_eq = optimconstr(N, N);
    % For indices where W(i,j)~=0, we enforce Z(i,j)>=0.
    consz_ineq = optimconstr(N, N);
    for i = 1:N
        for j = 1:N
            if W(i,j) == 0
                consz_eq(i,j) = Z(i,j) == 0;
            else
                consz_ineq(i,j) = Z(i,j) >= 0;
            end
        end
    end

    % A convexity constraint (inequality)
    consconv = norm(Z, 'fro')^2 <= 2 + a - 2*(1+a)/(gam*C5*sqrt(R));

    % Equality constraints for row sums
    consw1 = optimeq(N);
    for i = 1:N
        consw1(i) = sum(Z(i, :)) == 1;
    end

    % Equality constraints for column sums
    consw2 = optimeq(N);
    for i = 1:N
        consw2(i) = sum(Z(:, i)) == 1;
    end

    % --- Additional constraints based on indic, T_c, and T_thres ---
%    alpha = 0.01;
    alpha = 0.05;
    indic = alpha ./ (alpha + exp((-Z.^2) ./ alpha));
    T_c = (I .* S .* v) ./ f;
    L = indic .* gam .* T;

    % Build temperature constraints as a vector (one constraint per i,j)
    temps = optimconstr(N*N, 1);
    idx = 1;
    for i = 1:N
        for j = 1:N
            temps(idx) = L(i,j) + T_c(j) <= T_thres(i,j);
            idx = idx + 1;
        end
    end

    % Add all constraints to the problem
    prob.Constraints.consz_eq   = consz_eq;
    prob.Constraints.consz_ineq = consz_ineq;
    prob.Constraints.consconv   = consconv;
    prob.Constraints.consw1     = consw1;
    prob.Constraints.consw2     = consw2;
    prob.Constraints.temps      = temps;

    % --- Define the objective function ---
    E_c = sum(kap * I .* S .* v .* f.^2);
    obj = E_c + sum(sum(indic .* (p .* T .* gam))) + 0.5 * norm(W - Z, 'fro')^2;
    prob.Objective = obj;

    % Set initial guesses for the optimization variables
    x0.Z = W;
    x0.f = ones(N, 1) * 1e9
    x0.gam = 0.5;

    % --- Set timeout parameters and output function ---
    timeout = 200;
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
%        "StepTolerance", 1e-10, ...
%        "ConstraintTolerance", 1e-8, ...
%        "Algorithm", "interior-point", ...
    opts = optimoptions("fmincon", ...
        "Display", "none", ...
        "MaxIterations", 10000, ...
        "MaxFunctionEvaluations", 500000, ...
        "OutputFcn", @maxTimeFcn, ...
        "ScaleProblem", "obj-and-constr", ...
        "StepTolerance", 1e-6, ...
        "ConstraintTolerance", 1e-4, ...
        "Algorithm", "sqp", ...
        "ScaleProblem", true, ...
        "UseParallel", true);

    % Solve the optimization problem
    [sol, fval, eflag, output] = solve(prob, x0, Options=opts);

    fprintf('\n========================================\n');
    fprintf('Optimization Results:\n');
    fprintf('Current Round: %f\n', R);
    fprintf('C5: %f\n', C5);
    fprintf('A: %f\n', a);
    fprintf('Objective value: %f\n', fval);
    fprintf('Exit flag: %d\n', eflag);
    fprintf('Number of iterations: %d\n', output.iterations);
    fprintf('Number of function evaluations: %d\n', output.funcCount);
    fprintf('First-order optimality: %e\n', output.firstorderopt);
    fprintf('Execution time: %.2f seconds\n', toc(startTime));
    fprintf('========================================\n');


    % Check exit flag and use last feasible iterate if available
    if eflag <= 0
        if isfield(sol, 'Z') && isfield(sol, 'f') && isfield(sol, 'gam')
            % Apply thresholding to Z even if solver failed
            sol.Z(sol.Z < 0.09) = 0;
            A = sol.Z;
            B = sol.f;
            C = sol.gam;
            fprintf('Warning: Solver exited with flag %d. Returning last iterate.', eflag);
        else
            % No feasible solution found
            A = -999;
            B = -999;
            C = -999;
            fprintf('Error: No feasible iterate found.');
        end
    else
        % Successful convergence
        sol.Z(sol.Z < 0.09) = 0;
        A = sol.Z;
        B = sol.f;
        C = sol.gam;
    end
end
