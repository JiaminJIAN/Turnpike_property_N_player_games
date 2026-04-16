%==========================================================================
% Numerical illustration of turnpike properties in
% linear quadratic Gaussian N-player differential games
%
% Setting: d=1 (scalar), symmetric players
%   Reference positions: xbar_i^i = 1  (self-reference, non-zero)
%                        xbar_j^i = 0  for j ~= i
%   Initial mean:        mu_0^i   = 1  for all i
%
%
%   F_1^i(mu^{-i}_T(t)) = -Q_ii + B * sum_{j~=i} mu_j_T(t)
%
%   F_0^i(mu^{-i}_T(t), (Sigma^i_T)^{-1}) =
%       Q_ii - 2*(N-1)*B*mu_T(t)
%       + (N-1)*B / Sigma_T(t)
%       + (N-1)*B * (mu_T(t))^2
%
% Ergodic mean (by symmetry mu^i = mu* for all i):
%   mu* = 2*Q_ii / (R * A_cl^2 + 2*(N-1)*B)
%   rho* = R * A_cl * mu*
%
% Ergodic constant:
%   c^i = zeta*Lambda - (1/2)*rho*^2/R + F_0^i(mu*, Sigma_erg)
%==========================================================================

clear; close all; clc;
rng(42);

fprintf('==========================================================\n');
fprintf('Turnpike: LQG N-Player Games -- Non-Zero Reference\n');
fprintf('  xbar_i^i = 1,  xbar_j^i = 0 (j~=i),  mu0 = 1\n');
fprintf('==========================================================\n\n');

%==========================================================================
% SECTION 1: PARAMETERS
%==========================================================================
A_val    = -0.5;
sigma    =  0.8;
R_val    =  1.0;
Q_ii     =  1.0;
B_scale  =  0.5;    % B = B_scale/N  (O(1/N), Assumption 2.5)

mu0      =  1.0;    % common initial mean for all players
Sigma0   =  2.0;    % precision of initial distribution

T_long   = 20.0;
dt       =  0.02;
M        =  8000;   % Monte Carlo paths

Nt       = round(T_long/dt) + 1;
t_vec    = linspace(0, T_long, Nt);

N_vals          = [5, 10, 20, 100];
T_vals_corollary = [2, 4, 8, 12, 16, 20, 30, 50];

% Colours (blue, red, green, purple)
clrs = [0.122 0.471 0.706;
        0.839 0.153 0.157;
        0.173 0.627 0.173;
        0.580 0.404 0.741];

picard_tol     = 1e-8;
picard_maxiter = 500;

fprintf('A=%.2f, sigma=%.2f, R=%.2f, Q_ii=%.2f, B_scale=%.2f\n', ...
    A_val, sigma, R_val, Q_ii, B_scale);
fprintf('mu0=%.1f, Sigma0=%.1f, T=%.1f, dt=%.3f, M=%d\n\n', ...
    mu0, Sigma0, T_long, dt, M);

%==========================================================================
% SECTION 2: ERGODIC ALGEBRAIC RICCATI SOLUTIONS
%==========================================================================
zeta   = sigma^2 / 2;
Lambda = R_val * (A_val + sqrt(A_val^2 + 2*Q_ii/R_val));
A_cl   = A_val - (1/R_val)*Lambda;
lambda_tp = abs(A_cl);
Sigma_erg = ((1/R_val)*Lambda - A_val) / zeta;

fprintf('--- Ergodic ARE ---\n');
fprintf('  Lambda     = %.6f\n', Lambda);
fprintf('  A_cl       = %.6f\n', A_cl);
fprintf('  lambda_tp  = %.6f\n', lambda_tp);
fprintf('  Sigma_erg  = %.6f\n\n', Sigma_erg);

% Ergodic mu*, rho*, c^i for each N
%   mu* = 2*Q_ii / (R*A_cl^2 + 2*(N-1)*B)
%   rho* = R*A_cl*mu*
mu_star  = zeros(1, length(N_vals));
rho_star = zeros(1, length(N_vals));
c_erg    = zeros(1, length(N_vals));

fprintf('--- Ergodic Means and Values ---\n');
for kN = 1:length(N_vals)
    N = N_vals(kN);
    B = B_scale / N;
    mu_star(kN)  = 2*Q_ii / (R_val*A_cl^2 + 2*(N-1)*B);
    rho_star(kN) = R_val * A_cl * mu_star(kN);
    F0_erg = Q_ii - 2*(N-1)*B*mu_star(kN) ...
             + (N-1)*B/Sigma_erg ...
             + (N-1)*B*(mu_star(kN))^2;
    c_erg(kN) = zeta*Lambda - 0.5*(rho_star(kN))^2/R_val + F0_erg;
    fprintf('  N=%2d: B=%.4f, mu*=%.5f, rho*=%.5f, c^i=%.6f\n', ...
        N, B, mu_star(kN), rho_star(kN), c_erg(kN));
end

%==========================================================================
% SECTION 3: FINITE-HORIZON RICCATI ODES
%==========================================================================
% Lambda_T^i(t): backward ODE, Lambda_T(T)=0
%   dL/dt = -2*A*L + (1/R)*L^2 - 2*Q_ii
%
% Sigma_T^i(t): forward ODE, Sigma_T(0)=Sigma0
%   dS/dt = -2*zeta*S^2 - 2*S*(A - (1/R)*L)
%
% mu_T^i(t) and rho_T^i(t): coupled FB-ODE (system 52), Picard iteration
%   d/dt rho_T = -A_cl_T(t)*rho_T - 2*F_1^i,  rho_T(T)=0
%   d/dt mu_T  =  A_cl_T(t)*mu_T - (1/R)*rho_T, mu_T(0)=mu0
% where F_1^i = -Q_ii + (N-1)*B*mu_T  (with xbar_i^i=1, symmetric)

fprintf('\n--- Solving Finite-Horizon Riccati ODEs (T=%.1f) ---\n', T_long);

%--- Lambda_T (backward) ---
Lambda_T      = zeros(1, Nt);
for k = Nt-1:-1:1
    l         = Lambda_T(k+1);
    rhs       = -2*A_val*l + (1/R_val)*l^2 - 2*Q_ii;
    Lambda_T(k) = l - dt*rhs;
end

%--- Sigma_T (forward) ---
Sigma_T      = zeros(1, Nt);
Sigma_T(1)   = Sigma0;
for k = 1:Nt-1
    s          = Sigma_T(k);
    rhs        = -2*zeta*s^2 - 2*s*(A_val - (1/R_val)*Lambda_T(k));
    Sigma_T(k+1) = max(s + dt*rhs, 1e-12);
end

Acl_T_long = A_val - (1/R_val)*Lambda_T;   % time-varying A_cl_T(t)

fprintf('  Lambda_T(0)=%.5f [Lambda=%.5f],  Sigma_T(T)=%.5f [Sigma_erg=%.5f]\n', ...
    Lambda_T(1), Lambda, Sigma_T(end), Sigma_erg);

%--- mu_T and rho_T via Picard fixed-point (one solution per N) ---
fprintf('\n--- Picard Iteration for mu_T, rho_T ---\n');
mu_T_all  = zeros(length(N_vals), Nt);
rho_T_all = zeros(length(N_vals), Nt);

for kN = 1:length(N_vals)
    N  = N_vals(kN);
    B  = B_scale / N;
    mu_it  = mu0 * ones(1, Nt);
    rho_it = zeros(1, Nt);

    for iter = 1:picard_maxiter
        mu_old = mu_it;

        % Backward: rho_T(T)=0
        %   d/dt rho = -A_cl_T*rho - 2*F_1,  F_1 = -Q_ii + (N-1)*B*mu
        rho_new = zeros(1, Nt);
        for k = Nt-1:-1:1
            F1        = -Q_ii + (N-1)*B*mu_it(k+1);
            rhs       = -Acl_T_long(k+1)*rho_new(k+1) - 2*F1;
            rho_new(k) = rho_new(k+1) - dt*rhs;
        end

        % Forward: mu_T(0)=mu0
        %   d/dt mu = A_cl_T*mu - (1/R)*rho
        mu_new    = zeros(1, Nt);
        mu_new(1) = mu0;
        for k = 1:Nt-1
            rhs        = Acl_T_long(k)*mu_new(k) - (1/R_val)*rho_new(k);
            mu_new(k+1) = mu_new(k) + dt*rhs;
        end

        err    = max(abs(mu_new - mu_old));
        mu_it  = mu_new;
        rho_it = rho_new;
        if err < picard_tol, break; end
    end

    mu_T_all(kN,:)  = mu_it;
    rho_T_all(kN,:) = rho_it;
    fprintf('  N=%2d: iters=%3d | mu_T(0)=%.4f, rho_T(0)=%.5f, mu_T(T)=%.2e\n', ...
        N, iter, mu_it(1), rho_it(1), mu_it(end));
end

%==========================================================================
% SECTION 4: MONTE CARLO SIMULATION
%==========================================================================
% Finite-horizon SDE (19):
%   dX_T^j = [A_cl_T(t)*X_T^j - (1/R)*rho_T(t)] dt + sigma*dW^j
%
% Ergodic SDE (32):
%   dX^j = [A_cl*X^j - (1/R)*rho*] dt + sigma*dW^j
%
% Both driven by the SAME Brownian increments (Theorem 2.1).

fprintf('\n--- Monte Carlo Simulation (M=%d) ---\n', M);

E_sq_diff  = zeros(length(N_vals), Nt);
E_mu_T_mc  = zeros(length(N_vals), Nt);
E_mu_e_mc  = zeros(length(N_vals), Nt);

for kN = 1:length(N_vals)
    N       = N_vals(kN);
    rho_T_N = rho_T_all(kN,:);
    fprintf('  N=%2d ... ', N);

    X0 = mu0 + (1/sqrt(Sigma0)) * randn(M, 1);
    XT = X0;
    Xe = X0;

    sq = zeros(M, Nt);
    sq(:,1) = (XT - Xe).^2;
    E_mu_T_mc(kN,1) = mean(XT);
    E_mu_e_mc(kN,1) = mean(Xe);

    for k = 1:Nt-1
        dW = sqrt(dt) * randn(M, 1);

        XT = XT + (Acl_T_long(k)*XT - (1/R_val)*rho_T_N(k))*dt + sigma*dW;
        Xe = Xe + (A_cl*Xe - (1/R_val)*rho_star(kN))*dt + sigma*dW;

        sq(:,k+1) = (XT - Xe).^2;
        E_mu_T_mc(kN,k+1) = mean(XT);
        E_mu_e_mc(kN,k+1) = mean(Xe);
    end

    E_sq_diff(kN,:) = mean(sq, 1);
    fprintf('done.  max E[diff^2]=%.5f\n', max(E_sq_diff(kN,:)));
end

% Turnpike bound shape
tp_shape = exp(-lambda_tp*t_vec) + exp(-lambda_tp*(T_long - t_vec));

%==========================================================================
% SECTION 5: COROLLARY 2.2 -- VALUE FUNCTION ERGODICITY
%==========================================================================
% V_T^i(0,x) = (1/2)*x^2*Lambda_T(0) + rho_T(0)*x + kappa_T(0)
%
% kappa_T(0) = int_0^T [zeta*Lambda_T - (1/2)*rho_T^2/R + F_0^i] dt
% F_0^i = Q_ii - 2*(N-1)*B*mu_T + (N-1)*B/Sigma_T + (N-1)*B*(mu_T)^2

fprintf('\n--- Corollary 2.2: Value Function Ergodicity ---\n');

x_test   = mu0;
n_T      = length(T_vals_corollary);
V_over_T = zeros(length(N_vals), n_T);

for kN = 1:length(N_vals)
    N = N_vals(kN);
    B = B_scale / N;

    for kT = 1:n_T
        T_cur  = T_vals_corollary(kT);
        Nt_c   = round(T_cur/dt) + 1;
        t_c    = linspace(0, T_cur, Nt_c);

        % Lambda_T on [0,T_cur]
        LT_c = zeros(1, Nt_c);
        for k = Nt_c-1:-1:1
            l = LT_c(k+1);
            LT_c(k) = l - dt*(-2*A_val*l + (1/R_val)*l^2 - 2*Q_ii);
        end

        % Sigma_T on [0,T_cur]
        ST_c    = zeros(1, Nt_c);
        ST_c(1) = Sigma0;
        Acl_c   = A_val - (1/R_val)*LT_c;
        for k = 1:Nt_c-1
            s = ST_c(k);
            ST_c(k+1) = max(s + dt*(-2*zeta*s^2 - 2*s*(A_val-(1/R_val)*LT_c(k))), 1e-12);
        end

        % mu_T, rho_T via Picard
        mu_c  = mu0 * ones(1, Nt_c);
        rho_c = zeros(1, Nt_c);
        for iter = 1:picard_maxiter
            mu_old = mu_c;
            rho_new = zeros(1, Nt_c);
            for k = Nt_c-1:-1:1
                F1 = -Q_ii + (N-1)*B*mu_c(k+1);
                rhs = -Acl_c(k+1)*rho_new(k+1) - 2*F1;
                rho_new(k) = rho_new(k+1) - dt*rhs;
            end
            mu_new    = zeros(1, Nt_c);
            mu_new(1) = mu0;
            for k = 1:Nt_c-1
                mu_new(k+1) = mu_new(k) + dt*(Acl_c(k)*mu_new(k) - (1/R_val)*rho_new(k));
            end
            err   = max(abs(mu_new - mu_old));
            mu_c  = mu_new;
            rho_c = rho_new;
            if err < picard_tol, break; end
        end

        % kappa_T(0)
        F0_vec = Q_ii - 2*(N-1)*B*mu_c ...
                 + (N-1)*B./ST_c ...
                 + (N-1)*B*(mu_c).^2;
        kint   = zeta*LT_c - 0.5*(rho_c.^2)/R_val + F0_vec;
        kappa0 = trapz(t_c, kint);

        VT = 0.5*x_test^2*LT_c(1) + rho_c(1)*x_test + kappa0;
        V_over_T(kN,kT) = VT / T_cur;
    end

    fprintf('  N=%2d: (1/T)*V_T(T=50)=%.6f  [c^i=%.6f]\n', ...
        N, V_over_T(kN,end), c_erg(kN));
end

%==========================================================================
% SECTION 6: FIGURES (all separate)
%==========================================================================
fprintf('\n--- Generating Figures ---\n');

lw = 2.0;    % line width
fs = 12;     % font size
ms = 8;      % marker size

leg_str = arrayfun(@(n) sprintf('$N=%d$', n), N_vals, 'UniformOutput', false);

%--------------------------------------------------------------------------
% FIG 1: Non-uniform turnpike  (Theorem 2.1, Eq. 41)
%--------------------------------------------------------------------------
figure('Position',[100 100 700 460],'Color','w');
hold on;
for kN = 1:length(N_vals)
    plot(t_vec, E_sq_diff(kN,:), '-', 'Color', clrs(kN,:), 'LineWidth', lw, ...
         'DisplayName', leg_str{kN});
end
C = max(E_sq_diff(:)) / max(tp_shape);
plot(t_vec, C*tp_shape, 'k--', 'LineWidth', lw, ...
     'DisplayName', sprintf('Bound'));
xlabel('Time $t$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('$E[|X^i_T(t)-X^i(t)|^2]$', 'Interpreter','latex', 'FontSize', fs+1);
title({'Turnpike property'}, 'Interpreter','latex', 'FontSize', fs);
legend('Location','north', 'Interpreter','latex', 'FontSize', fs-1);
xlim([0 T_long]);
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);

%--------------------------------------------------------------------------
% FIG 2: Uniform-in-N turnpike  (Theorem 2.1, Eq. 42)
%--------------------------------------------------------------------------
figure('Position',[120 100 700 460],'Color','w');
hold on;
Cu = max(E_sq_diff(:)) / max(tp_shape) * 1.05;
for kN = 1:length(N_vals)
    plot(t_vec, E_sq_diff(kN,:), '-', 'Color', clrs(kN,:), 'LineWidth', lw, ...
         'DisplayName', leg_str{kN});
end
plot(t_vec, Cu*tp_shape, 'k--', 'LineWidth', lw+0.2, ...
     'DisplayName', 'Uniform bound');
xlabel('Time $t$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('$\frac{1}{N} E[|X_T(t)-X(t)|^2]$', 'Interpreter','latex', 'FontSize', fs+1);
title('Uniform-in-$N$ turnpike property', ...
      'Interpreter','latex', 'FontSize', fs);
legend('Location','north', 'Interpreter','latex', 'FontSize', fs-1);
xlim([0 T_long]); % grid on; 
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);

%--------------------------------------------------------------------------
% FIG 3: mu_T^i(t) -> mu*  (Proposition 2.3, Eq. 35)
%--------------------------------------------------------------------------
figure('Position',[140 100 700 460],'Color','w');
hold on;
for kN = 1:length(N_vals)
    plot(t_vec, mu_T_all(kN,:), '-', 'Color', clrs(kN,:), 'LineWidth', lw, ...
         'DisplayName', leg_str{kN});
    yline(mu_star(kN), '--', 'Color', clrs(kN,:), 'LineWidth', 1.2, 'Alpha', 0.75, 'HandleVisibility','off');
end
plot(NaN, NaN, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Ergodic $\mu^i$');
xlabel('Time $t$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('$\mu^i_T(t)$ and $\mu^i$', 'Interpreter','latex', 'FontSize', fs+1);
title({'Convergence of $\mu^i_T(t)$ to $\mu^i$'}, ...
      'Interpreter','latex', 'FontSize', fs);
legend('Location','northeast', 'Interpreter','latex', 'FontSize', fs-1);
% grid on; 
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);

%--------------------------------------------------------------------------
% FIG 4: rho_T^i(t) -> rho^i  (Proposition 2.3, Eq. 35)
%--------------------------------------------------------------------------
figure('Position',[160 100 700 460],'Color','w');
hold on;
for kN = 1:length(N_vals)
    plot(t_vec, rho_T_all(kN,:), '-', 'Color', clrs(kN,:), 'LineWidth', lw, ...
         'DisplayName', leg_str{kN});
    yline(rho_star(kN), '--', 'Color', clrs(kN,:), 'LineWidth', 1.2, 'Alpha', 0.75, 'HandleVisibility','off');
end
plot(NaN, NaN, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Ergodic $\rho^i$');
xlabel('Time $t$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('$\rho^i_T(t)$ and $\rho^i$', 'Interpreter','latex', 'FontSize', fs+1);
title({'Convergence of $\rho^i_T(t)$ to $\rho^i$'}, ...
      'Interpreter','latex', 'FontSize', fs);
legend('Location','southeast', 'Interpreter','latex', 'FontSize', fs-1);
ylim([-1.1, 0]);
% grid on; 
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);
%saveas(gcf, 'fig4_rho_convergence.png');
%fprintf('  Fig 4 saved.\n');

%--------------------------------------------------------------------------
% FIG 5: Lambda_T^i(t) -> Lambda^i  (Proposition 2.3, Eq. 33)
%--------------------------------------------------------------------------
figure('Position',[180 100 700 460],'Color','w');
hold on;
plot(t_vec, Lambda_T, 'b-', 'LineWidth', lw, 'DisplayName', '$\Lambda^i_T(t)$');
yline(Lambda, 'r--', 'LineWidth', lw, ...
      'Label', sprintf('$\\Lambda^i=%.4f$', Lambda), ...
      'Interpreter', 'latex', 'LabelHorizontalAlignment', 'right', ...
      'FontSize', fs-1, 'HandleVisibility','off');
plot(NaN, NaN, 'r--', 'LineWidth', lw, 'DisplayName', sprintf('$\\Lambda^i=%.4f$', Lambda));
xlabel('Time $t$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('$\Lambda^i_T(t)$ and $\Lambda^i$', 'Interpreter','latex', 'FontSize', fs+1);
title({'Convergence of $\Lambda^i_T(t)$ to $\Lambda^i$'}, ...
      'Interpreter','latex', 'FontSize', fs);
legend('Location','southeast', 'Interpreter','latex', 'FontSize', fs-1);
ylim([0 1.1]); 
% grid on; 
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);

%--------------------------------------------------------------------------
% FIG 6: Sigma_T^i(t) -> Sigma^i  (Proposition 2.3, Eq. 34)
%--------------------------------------------------------------------------
figure('Position',[190 100 700 460],'Color','w');
hold on;
plot(t_vec, Sigma_T, 'b-', 'LineWidth', lw, 'DisplayName', '$\Sigma^i_T(t)$');
yline(Sigma_erg, 'r--', 'LineWidth', lw, ...
      'Label', sprintf('$\\Sigma^i=%.4f$', Sigma_erg), ...
      'Interpreter', 'latex', 'LabelHorizontalAlignment', 'right', ...
      'FontSize', fs-1, 'HandleVisibility', 'off');
plot(NaN, NaN, 'r--', 'LineWidth', lw, ...
     'DisplayName', sprintf('$\\Sigma^i=%.4f$', Sigma_erg));
xlabel('Time $t$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('$\Sigma^i_T(t)$ and $\Sigma^i$', 'Interpreter','latex', 'FontSize', fs+1);
title({'Convergence of $\Sigma^i_T(t)$ to $\Sigma^i$'}, ...
      'Interpreter','latex', 'FontSize', fs);
legend('Location','southeast', 'Interpreter','latex', 'FontSize', fs-1);
% grid on; 
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);

%--------------------------------------------------------------------------
% FIG 7: MC mean paths E[X_T^j] vs E[X^j]
%--------------------------------------------------------------------------
figure('Position',[200 100 700 460],'Color','w');
hold on;
for kN = 1:length(N_vals)
    plot(t_vec, E_mu_T_mc(kN,:), '-',  'Color', clrs(kN,:), 'LineWidth', lw, 'HandleVisibility','off');
    plot(t_vec, E_mu_e_mc(kN,:), '--', 'Color', clrs(kN,:), 'LineWidth', 1.3, 'HandleVisibility','off');
end
% Legend proxies
for kN = 1:length(N_vals)
    plot(NaN, NaN, '-', 'Color', clrs(kN,:), 'LineWidth', lw, 'DisplayName', leg_str{kN});
end
plot(NaN, NaN, 'k-',  'LineWidth', lw,  'DisplayName', 'Finite-horizon mean path');
plot(NaN, NaN, 'k--', 'LineWidth', 1.3, 'DisplayName', 'Ergodic mean path');
xlabel('Time $t$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('Mean state', 'Interpreter','latex', 'FontSize', fs+1);
title({'Monte Carlo mean paths'}, ...
      'Interpreter','latex', 'FontSize', fs);
legend('Location','northeast', 'Interpreter','latex', 'FontSize', fs-2);
xlim([0 T_long]); 
% grid on; 
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);

%--------------------------------------------------------------------------
% FIG 8: Mean deviation |E[X_T^j] - E[X^j]|
%--------------------------------------------------------------------------
figure('Position',[220 100 700 460],'Color','w');
hold on;
for kN = 1:length(N_vals)
    plot(t_vec, abs(E_mu_T_mc(kN,:) - E_mu_e_mc(kN,:)), '-', ...
         'Color', clrs(kN,:), 'LineWidth', lw, 'DisplayName', leg_str{kN});
end
xlabel('Time $t$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('$|E[X^j_T(t)]-E[X^j(t)]|$', 'Interpreter','latex', 'FontSize', fs+1);
title({'Turnpike in mean'}, ...
      'Interpreter','latex', 'FontSize', fs);
legend('Location','north', 'Interpreter','latex', 'FontSize', fs-1);
xlim([0 T_long]); % grid on; 
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);

%--------------------------------------------------------------------------
% FIG 9: Corollary 2.2 -- (1/T)*V_T^i -> c^i
%--------------------------------------------------------------------------
figure('Position',[240 100 700 460],'Color','w');
hold on;
T_arr = T_vals_corollary;
for kN = 1:length(N_vals)
    plot(T_arr, V_over_T(kN,:), 'o-', 'Color', clrs(kN,:), 'LineWidth', lw, ...
         'MarkerSize', ms, 'MarkerFaceColor', clrs(kN,:), ...
         'DisplayName', sprintf('$N=%d$,\\ $c^i=%.4f$', N_vals(kN), c_erg(kN)));
    yline(c_erg(kN), '--', 'Color', clrs(kN,:), 'LineWidth', 1.0, 'Alpha', 0.7, 'HandleVisibility','off');
end
xlabel('Time horizon $T$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('$\frac{1}{T}V^i_T$ and $c^i$', 'Interpreter','latex', 'FontSize', fs+1);
title({'Convergence of $\frac{1}{T}V^i_T$ to $c^i$'}, ...
      'Interpreter','latex', 'FontSize', fs);
legend('Location','southeast', 'Interpreter','latex', 'FontSize', fs-1);
xlim([0, max(T_vals_corollary)+3]); 
ylim([0.52, 0.58]);
% grid on; 
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);

%--------------------------------------------------------------------------
% FIG 10: Ratio test over time -- uniform-in-N
%--------------------------------------------------------------------------
figure('Position',[260 100 700 460],'Color','w');
hold on;
idx = round(Nt*0.05):round(Nt*0.95);
mr  = zeros(1, length(N_vals));
for kN = 1:length(N_vals)
    r = E_sq_diff(kN,:) ./ (tp_shape + 1e-12);
    plot(t_vec(idx), r(idx), '-', 'Color', clrs(kN,:), 'LineWidth', lw, ...
         'DisplayName', leg_str{kN});
    mr(kN) = max(r(idx));
end
yline(max(mr)*1.1, 'r--', 'LineWidth', lw, ...
      'Label', '$\tilde{K}$', 'Interpreter','latex', ...
      'LabelHorizontalAlignment','right', 'FontSize', fs, 'HandleVisibility','off');
xlabel('Time $t$', 'Interpreter','latex', 'FontSize', fs+1);
ylabel('Ratio', ...
       'Interpreter','latex', 'FontSize', fs+1);
title({'Ratio test for uniform-in-$N$ bound'}, ...
      'Interpreter','latex', 'FontSize', fs);
legend('Location','north', 'Interpreter','latex', 'FontSize', fs-1);
% grid on; 
box on; 
set(gca,'FontSize',fs,'LineWidth',1.2);

%--------------------------------------------------------------------------
% FIG 11: Max ratio vs N
%--------------------------------------------------------------------------
%figure('Position',[280 100 620 440],'Color','w');
%hold on;
%plot(N_vals, mr, 'ks-', 'LineWidth', lw, 'MarkerSize', ms+2, ...
%     'MarkerFaceColor', 'k', 'DisplayName', '$\max_t$ ratio');
%yline(max(mr)*1.1, 'r--', 'LineWidth', lw, ...
%      'Label', 'Interpreter','latex', ...
%      'LabelHorizontalAlignment','right', 'FontSize', fs);
%plot(NaN, NaN, 'r--', 'LineWidth', lw, ...
%     'DisplayName', '$\tilde{K}$ (uniform bound)');
%xlabel('Number of players $N$', 'Interpreter','latex', 'FontSize', fs+1);
%ylabel('$\max_t$ ratio', 'Interpreter','latex', 'FontSize', fs+1);
%title({'Uniform bound constant'}, 'Interpreter','latex', 'FontSize', fs);
%legend('Location','northeast', 'Interpreter','latex', 'FontSize', fs-1);
%xticks(N_vals); % grid on; 
%box on; 
%set(gca,'FontSize',fs,'LineWidth',1.2);

%==========================================================================
% SUMMARY
%==========================================================================
fprintf('\n==========================================================\n');
fprintf('SUMMARY\n');
fprintf('==========================================================\n');
fprintf('Lambda=%.4f, A_cl=%.4f, lambda_tp=%.4f, Sigma_erg=%.4f\n\n', ...
    Lambda, A_cl, lambda_tp, Sigma_erg);
fprintf('%-5s %-8s %-10s %-10s %-12s\n','N','B','mu*','rho*','c^i');
fprintf('%s\n', repmat('-',1,50));
for kN = 1:length(N_vals)
    fprintf('%-5d %-8.4f %-10.5f %-10.5f %-12.6f\n', ...
        N_vals(kN), B_scale/N_vals(kN), mu_star(kN), rho_star(kN), c_erg(kN));
end
fprintf('\nAll 10 figures saved to current directory.\n');
