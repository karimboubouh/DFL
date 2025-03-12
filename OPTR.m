function [A, B, C] = OPT_2(W,R)

% Suppress all warnings
warning('off', 'all');

%%% this c_k
v= [1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;1.5*10^4;1.5*10^4;10^4;10^4;3*10^4];
%f=2*10^10*[0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1];
%c= [10^4;2*10^4;3*10^4;1.5*10^4;10^4;2*10^4;3*10^4;1.5*10^4;10^4;2*10^4;3*10^4;1.5*10^4;10^4;2*10^4;3*10^4;1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;10^4;3*10^4;2.5*10^4;1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;10^4;3*10^4;2.5*10^4;1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;10^4;3*10^4;2.5*10^4;1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;10^4;2*10^4;3*10^4;1.5*10^4;10^4;2*10^4;3*10^4;1.5*10^4;10^4;2*10^4;3*10^4;1.5*10^4;10^4;2*10^4;3*10^4;1.5*10^4;10^4;2.3*10^4;1.7*10^4;1.2*10^4;3*10^4;1.5*10^4;10^4;2.3*10^4;1.7*10^4;1.2*10^4;1.7*10^4;1.2*10^4;2.5*10^4;1.5*10^4;10^4;10^4;3*10^4;2.5*10^4;1.5*10^4;10^4];
%% this is D_k
S=[700;30;20;80;120;600;2000;39;459;340];
kap=10^-28;
I=1;
p=10*[0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1];
%p = 10 * [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
%band = 2*10^6*[0.1;1;10;0.1;0.1;0.1;1;0.1;0.1;0.1];
band = 2*10^7*[0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1];
dim = 500000;

T_thres= [1.2500   27.4152    1.0110    50.8624    1.0216    1.0967    1.0375    1.0221    1.0375    1.0375;
    1.0142    1.2500    1.0569    1.0833    1.0355    1.0569    1.0327    1.1577    1.0327    1.0569;
    1.0967    1.0569    1.2500    1.1577    1.1577    1.0789    1.0625    1.1577    1.1577    1.1577;
    1.0199    1.1577    1.0324    1.2500    1.1577    1.1577    1.1577    1.0891    1.0279    1.0967;
    1.0723    1.0569    1.0789    10.8181   10.2500    1.2500    1.1577    1.0327    1.1577    1.1577;
    1.0789    1.2500    1.1577    10.8181    10.1577    1.2500    1.0327    1.0723    1.0279    1.2500;
    1.0723    1.0318    1.1577    1.1577   117.4152    1.1577    1.2500    1.2500   117.4152    1.1577;
    1.0723    1.0789    1.1577    1.0789    1.1577    1.0789    1.0569    1.2500    1.1577    1.1577;
    1.0569    1.0327    1.1250    1.1577    1.1077    1.0789    1.0723    1.0327    1.2500    1.1077;
    1.0259    1.0789    1.1577    1.1577    1.1577    1.1577    1.0327    1.1577    1.1250    1.2500];

T= [0	30	0.00007	0.3	0.03	0.005	0.01	0.002	0.01	0.01;
0.02	0	10	0.07	0.13	0.02	0.02	0.02	0.02	0.002;
0.05	0.02	0	0.02	0.02	0.8	15	8	0.02	0.02;
0.006	0.02	5	0	0.02	0.02	0.02	6	0.05	0.005;
0.00001	0.02	0.008	0.1	0	15	0.02	0.00002	0.02	30;
0.008	0.1	0.02	0.1	0.02	0	0.02	0.1	15	10;
0.01	0.02	0.02	0.02	10	0.02	0	0	10	0.02;
0.01	0.008	0.02	0.008	0.02	8	0.2	0	0.02	0.002;
0	0.02	0.003	0.02	0.004	0.008	0.1	0.02	0	0.004;
25	0.00008	200	0.02	0.02	0.02	0.02	0.02	0.003	0];


%C5=10;
C5=1;
%R=10;
a=3;


zer = zeros(10, 10);


oness = ones(10, 10);
d= 0.5;


N=10;
X=1;

prob = optimproblem;
%creating the optimization variables
Z = optimvar('Z',N,N,'LowerBound',zer,'UpperBound',oness);
gam = optimvar('gam',1,'LowerBound',0.1,'UpperBound',0.9);
f = optimvar('f',N,'LowerBound',10^6,'UpperBound',10^9);

%now lets define the constraints
consz = optimeq(N,N);
consz = [];
for i = 1:N
    for j = 1:N     
        if W(i,j) == 0
            consz = [consz, Z(i,j) == 0];
        end
    end
end
consconv = optimineq(1);

consconv =  norm(Z, 'fro')^2 <=  2 + a - 2*(1+a)/(gam*C5*sqrt(R));
%1 - (2 /(C5 * sqrt(R)))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

consw1 = optimeq(N);
for i=1:N
    consw1(i) = sum(Z(i, :)) == 1 ;
end

consw2 = optimeq(N);
for i=1:N
    consw2(i) = sum(Z(:, i)) == 1 ;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha= 10^-2;
indic = alpha./(alpha+ exp((-Z.^2)./alpha));
temps = optimineq(N,N);

T_c = (I .* S .* v) ./f;
temps = [];
%L=Z.* gam.*T;
L=indic.* gam.*T;
for i = 1:N
    for j = 1:N     
        temps = [temps, L(i,j)+ T_c(j) <= T_thres(i,j)] ;
    end
end

prob.Constraints.consz=consz;
prob.Constraints.consconv=consconv;
prob.Constraints.consw1=consw1;
prob.Constraints.consw2=consw2;
prob.Constraints.temps=temps;

%define the objective

E_c = sum(kap * I .* S .* v .* f.^2);


obj = E_c + sum( sum(indic.*(p.*T.*gam)) )  +   0.5*norm(W-Z, 'fro')^2;

%obj = E_c + sum( sum(Z.*(T .* repmat(p, 1, 10))) )  +   0.5*norm(W-Z, 'fro')^2;

prob.Objective=obj;
%initialisation
%x0.Z=zer;
x0.Z=W;
x0.f=ones(N,1)*10^6;
x0.gam=0.25;


%opts=optimoptions("fmincon",Display="iter",MaxIterations=10000,PlotFcn={@optimplotx, @optimplotfval},MaxFunctionEvaluations = 300000, ScaleProblem=true);

timeout = 280;
startTime = tic;
function stop = maxTimeFcn(~,optimValues,~)
    elapsedTime = toc(startTime);
    if elapsedTime > timeout
        fprintf('Timeout reached: Stopping optimization at iteration %d.\n', optimValues.iteration);
        stop = true;
    else
        stop = false;
    end
end

%    Display="iter", ...
%    PlotFcn={@optimplotx, @optimplotfval}, ...
%    UseParallel=true,...
opts = optimoptions("fmincon", ...
    Display="none", ...
    MaxIterations=10000, ...
    MaxFunctionEvaluations=500000, ...
    OutputFcn=@maxTimeFcn, ...
    StepTolerance=1e-10, ...
    ConstraintTolerance=1e-8, ...
    Algorithm='interior-point', ...
    ScaleProblem=true);



%opts=optimoptions("fmincon",Display="iter",MaxIterations=10000,PlotFcn={@optimplotx, @optimplotfval},MaxFunctionEvaluations = 10000, ScaleProblem=true);
[sol,fval,eflag,output] = solve(prob,x0,Options=opts);

% Results
fprintf('\n========================================\n');
fprintf('Optimization Results:\n');
fprintf('Current Round: %f\n', R);
fprintf('Objective value: %f\n', fval);
fprintf('Exit flag: %d\n', eflag);
fprintf('Number of iterations: %d\n', output.iterations);
fprintf('Number of function evaluations: %d\n', output.funcCount);
fprintf('First-order optimality: %e\n', output.firstorderopt);
fprintf('Execution time: %.2f seconds\n', toc(startTime));
fprintf('========================================\n');

if eflag == -1
    A = -999;
    B = -999;
    C = -999;
else
    sol.Z(sol.Z < 0.09) = 0;
    A = sol.Z;  % Works for optimization variables
    B = sol.f;
    C = sol.gam;
end

end