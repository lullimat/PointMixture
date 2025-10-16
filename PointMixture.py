import numpy as np
import sympy as sp

"""
Workflow:
1. Input -> weights 'w_i', abscissas 'hat_x_i' and likelihood matrix 'L_ij'
2. Define the moment function given the quadrature
3. Compute the standardized moments correlation matrix 'Cpq'
4. Compute the Moment-Orthogonal polynomials evaluations 'q_poly_evals'/'q_poly_evals_T'
5. Find 'beta_max' and set 'beta = beta_max * 0.1'
6. Compute the correlation matrix for the moments: 'sqrt_beta_scale', 'Cpq_beta = sqrt_beta_scale @ (Cpq @ sqrt_beta_scale)'
7. Use 'GetPDFCDFLogW(Hm1_T, Cpq_beta, q_poly_evals, w_i, max_cond=1e10)' to get 'ranges_list, pdf_list, cdf_list', i.e. log(w) 
    ranges, pdf and cdf obtained from the convolution of generalized non-centered $\chi^2$ distributions
8. Get the log
8. Find the optimal weights that maximize the likelihood with the function 'find_optimal_w' passing 
    'L_ij', 'pdf_list', 'cdf_list', 'ranges_list', 'corr_logWlogW' as parameters
9. Check if the optimized weights are sizeably different from the initial one: 
    - if yes, substitute the optimized weights to 'w_i' and restart from point 1.
    - if no, declare convergence and return the weights
"""

def GetCpq(P, ms_lam):
    ms_dict_P = {p: ms_lam(p) for p in range(1, 2 * P + 1, 1)}
    num_lam = lambda p, q: ms_dict_P[p + q] - ms_dict_P[p] * ms_dict_P[q]
    den_lam = lambda p, q: np.sqrt((ms_dict_P[2 * p] - (ms_dict_P[p] ** 2)) * (ms_dict_P[2 * q] - (ms_dict_P[q] ** 2)))
    Cpq = np.array([[num_lam(p, q) / den_lam(p, q) for q in range(1, P + 1)] for p in range(1, P + 1)])
    return Cpq

def GetQuadMomentOrthogonalPolynomials(x_hat, quad_ms):
    q_degree = len(x_hat) - 1
    H = np.array([[quad_ms(i + j) for j in range(q_degree + 1)] for i in range(q_degree + 1)])
    Hm1_T = np.linalg.inv(H).T

    q_polys_c = np.array([Hm1_T @ np.array([1 if j == i else 0 for j in range(q_degree + 1)]) for i in range(q_degree + 1)])
    q_polys_c_np = np.array([np.flip(cs) for cs in q_polys_c])

    q_polys_lams = [lambda x, c=cs: np.polyval(c, x) for cs in q_polys_c_np]
    q_poly_evals = np.array([f(x_hat) for f in q_polys_lams])
    return Hm1_T, q_poly_evals
    
def FindMaxBeta(P, ms_lam, GL_ratio=0.1):
    ratios_mm2 = np.array([ms_lam(p) / np.sqrt((ms_lam(2 * p) - (ms_lam(p) ** 2))) for p in range(1, P + 1)])
    ratios_mm2 = ratios_mm2[ratios_mm2 > 0]
    kBT_min = 1 / np.sort(ratios_mm2)[0]
    beta_max = (1 / kBT_min) * GL_ratio
    return beta_max

def SqrtVarBeta(p, ms_lam, beta=1):
    return np.sqrt(beta * (ms_lam(2 * p) - (ms_lam(p) ** 2)))

from scipy.signal import fftconvolve
from scipy.stats import ncx2
import scipy.integrate as integrate
from scipy.integrate import cumulative_trapezoid as cumtrapz

from scipy.interpolate import UnivariateSpline
def GeneralizedNonCenteredChi2(lambda_i, mu_i):
    df, nc_i = 1, (mu_i ** 2)

    # Define lower bound using 0.01 percentile
    lower_list = [ncx2.ppf(0.0001, df=df, nc=nc_i[i], scale=lambda_i[i]) for i in range(len(mu_i))]
    lower = min(lower_list)  # Use the minimum to cover all distributions
    if lower < 1e-4:
        lower = 0  # Ensure non-negative

    # Compute high percentiles for a robust upper bound
    upper_list = [ncx2.ppf(0.9999, df=df, nc=nc_i[i], scale=lambda_i[i]) for i in range(len(mu_i))]
    upper = max(upper_list) * 1.1  # Add margin for tails

    # Unified grid with more points for resolution
    num_points = 2 ** 12  # 4096 points for better accuracy
    common_range = np.linspace(0, upper, num_points)
    dx = common_range[1] - common_range[0]

    # Recompute PDFs on the common grid
    chi2_list = [ncx2.pdf(common_range, df=df, nc=nc_i[i], scale=lambda_i[i]) for i in range(len(mu_i))]

    convolution = chi2_list[0]
    for chi2 in chi2_list[1:]:
        convolution = fftconvolve(convolution, chi2, mode='full') * dx

    # Grid for the final convolved PDF (sum ranges from 0 to ~3*upper)
    conv_range = np.linspace(0, common_range[-1] * len(mu_i), len(convolution))

    # Check integral (should be ~1)
    integral = integrate.trapezoid(convolution, conv_range)
    # print(f"Convolved PDF integral: {integral:.4f}")  # Adjust if <0.99 or >1.01
    if np.abs(1 - integral) > 0.01:
        convolution /= integral
        # print("New norm:", integrate.trapezoid(convolution, conv_range))

    pdf_spl = UnivariateSpline(conv_range, convolution, s=0, ext=3)
    
    # Compute CDF using cumulative integration
    cdf = cumtrapz(convolution, conv_range, initial=0)
    # Ensure CDF is non-decreasing and bounded at 1
    cdf = np.clip(cdf, 0, 1)
    # Create spline for CDF
    cdf_spl = UnivariateSpline(conv_range, cdf, s=0, ext=3)
    conv_range = np.linspace(np.sum(lower_list), np.sum(upper_list), 2 ** 10)
    return conv_range, pdf_spl, cdf_spl

from scipy.interpolate import UnivariateSpline
def GetPDFCDFLogW(Hm1_T, Cpq_beta, q_poly_evals, w_i, max_cond=1e10):
    # Ensure Cpq_beta is positive semi-definite
    if not np.all(np.linalg.eigvals(Cpq_beta) >= -1e-10):
        Cpq_beta = (Cpq_beta + Cpq_beta.T) / 2  # Symmetrize
        Cpq_beta = Cpq_beta + np.eye(Cpq_beta.shape[0]) * 1e-10  # Add small diagonal

    A = Hm1_T[1:, 1:]
    if np.linalg.cond(A) > max_cond:
        raise ValueError("Matrix A is ill-conditioned, consider regularization")
    Am1 = np.linalg.inv(A)

    # Eigenvalue decompositions with sorting
    Lambda, Q = np.linalg.eigh(A @ Cpq_beta)
    idx_sort = np.argsort(np.abs(Lambda))[::-1]
    Lambda = Lambda[idx_sort].real  # Take real part to avoid numerical artifacts
    Q = Q[:, idx_sort]

    D, V = np.linalg.eig(Cpq_beta)
    idx_sort_D = np.argsort(D)[::-1]
    D = D[idx_sort_D].real
    V = V[:, idx_sort_D]
    S = V @ np.diag(np.sqrt(np.maximum(D, 0))) @ V.T
    Sm1 = V @ np.diag(1 / np.sqrt(np.maximum(D, 1e-10))) @ V.T

    ATilda = S.T @ A @ S
    Gamma, P = np.linalg.eigh(ATilda)
    idx_sort_G = np.argsort(Gamma)[::-1]
    Gamma = Gamma[idx_sort_G].real
    P = P[:, idx_sort_G]

    # Transform nu_i to the eigenbasis of ATilda
    ranges_list, pdf_list, cdf_list = [], [], []
    for idx in range(len(w_i)):
        b_i = q_poly_evals[1:, idx]
        mu_i = Am1 @ b_i
        nu_i = -mu_i

        eta_i = P.T @ (Sm1 @ nu_i)
        eta_i = np.where(np.abs(eta_i) < 1e-6, 1e-6, eta_i)
        d_i = 0.5 * b_i @ Am1 @ b_i + np.log(w_i[idx])

        conv_range, pdf_spl, cdf_spl = GeneralizedNonCenteredChi2(Gamma, eta_i)
        
        log_w_range = d_i-conv_range/2
        log_w_pdf_spl = UnivariateSpline(np.flip(log_w_range), np.flip(2 * pdf_spl(conv_range)), s=0, ext=3)
        log_w_cdf_spl = UnivariateSpline(np.flip(log_w_range), 1 - cdf_spl(np.flip(conv_range)), s=0, ext=3)
        log_w_range = np.flip(log_w_range)
        
        ranges_list += [log_w_range]
        pdf_list += [log_w_pdf_spl]
        cdf_list += [log_w_cdf_spl]
        
    return ranges_list, pdf_list, cdf_list


def GetLogWCorrelations(Hm1_T, Cpq_beta, q_poly_evals, w_i, max_cond=1e10):
    # Ensure Cpq_beta is positive semi-definite
    if not np.all(np.linalg.eigvals(Cpq_beta) >= -1e-10):
        Cpq_beta = (Cpq_beta + Cpq_beta.T) / 2  # Symmetrize
        Cpq_beta = Cpq_beta + np.eye(Cpq_beta.shape[0]) * 1e-10  # Add small diagonal

    A = Hm1_T[1:, 1:]
    if np.linalg.cond(A) > max_cond:
        raise ValueError("Matrix A is ill-conditioned, consider regularization")
    Am1 = np.linalg.inv(A)

    # Eigenvalue decompositions with sorting
    Lambda, Q = np.linalg.eigh(A @ Cpq_beta)
    idx_sort = np.argsort(np.abs(Lambda))[::-1]
    Lambda = Lambda[idx_sort].real  # Take real part to avoid numerical artifacts
    Q = Q[:, idx_sort]

    D, V = np.linalg.eig(Cpq_beta)
    idx_sort_D = np.argsort(D)[::-1]
    D = D[idx_sort_D].real
    V = V[:, idx_sort_D]
    S = V @ np.diag(np.sqrt(np.maximum(D, 0))) @ V.T
    Sm1 = V @ np.diag(1 / np.sqrt(np.maximum(D, 1e-10))) @ V.T

    ATilda = S.T @ A @ S
    Gamma, P = np.linalg.eigh(ATilda)
    idx_sort_G = np.argsort(Gamma)[::-1]
    Gamma = Gamma[idx_sort_G].real
    P = P[:, idx_sort_G]

    # Transform nu_i to the eigenbasis of ATilda
    eta_i_list = []
    for idx in range(len(w_i)):
        b_i = q_poly_evals[1:, idx]
        mu_i = Am1 @ b_i
        nu_i = -mu_i

        eta_i = P.T @ (Sm1 @ nu_i)
        eta_i = np.where(np.abs(eta_i) < 1e-6, 1e-6, eta_i)
        eta_i_list += [eta_i]
        
    eta_i_list = np.array(eta_i_list)

    cov_YY = [[np.sum((Lambda ** 2) * (2 + 4 * eta_i_list[i, :] * eta_i_list[j, :])) for j in range(len(w_i))] for i in range(len(w_i))]
    cov_logWlogW = np.array(cov_YY) / 4

    corr_logWlogW = np.array([[cov_logWlogW[i, j] / np.sqrt(cov_logWlogW[i, i] * cov_logWlogW[j, j]) 
                               for j in range(len(w_i))] 
                              for i in range(len(w_i))])

    return cov_logWlogW, corr_logWlogW


from scipy.optimize import minimize
from scipy.stats import norm

def find_optimal_w(L_ij, pdf_list, cdf_list, ranges_list, corr_logWlogW, hat_w_sample=None, 
                   initial_w=None, constrained=True, regularization=1e-6, solver='trust-constr', debug=False, maxiter=2**11):
    """
    Finds the maximum of the log-posterior distribution for weights w in a point mixture model.

    The log-posterior is proportional to:
        sum_{i=1}^N log( sum_{j=1}^n L_{ij} w_j ) + log c_log( F_1^{(log)}(log w_1), ..., F_n^{(log)}(log w_n) )
        + sum_{i=1}^n log p_i(log w_i) - sum_{i=1}^n log w_i

    Parameters:
    - L_ij: Likelihood matrix, shape (N, n).
    - pdf_list: List of spline functions for p_i(log w_i).
    - cdf_list: List of spline functions for F_i^{(log)}(log w_i).
    - ranges_list: List of arrays defining valid ranges for log w_i.
    - corr_logWlogW: Correlation matrix for the Gaussian copula.
    - hat_w_sample: Sampled weights for initial guess (optional).
    - initial_w: Initial guess for w (default: uniform 1/n or mean of hat_w_sample).
    - constrained: If True, enforces sum(w) = 1 using softmax. If False, optimizes with w > 0.
    - regularization: Value added to diagonal of corr_logWlogW.
    - solver: Optimization method ('trust-constr' or 'SLSQP' for constrained case).
    - debug: If True, prints intermediate values.

    Returns:
        optimal_w: Optimized weights (normalized if unconstrained).
        log_post_value: Log-posterior value at the optimum.
        sum_w: Sum of the optimized weights.
    """
    N, n = L_ij.shape
    if initial_w is None:
        if hat_w_sample is not None:
            initial_w = np.mean(hat_w_sample, axis=0)
        else:
            initial_w = np.full(n, 1.0 / n)
    initial_w = np.array(initial_w) / np.sum(initial_w)  # Normalize

    # Regularize correlation matrix
    P = corr_logWlogW + np.eye(n) * regularization
    if not np.all(np.linalg.eigvals(P) > 0):
        raise ValueError("Regularized correlation matrix is not positive definite")

    # Precompute P inverse and determinant
    try:
        P_inv = np.linalg.inv(P)
        det_P = np.linalg.det(P)
        if det_P <= 0:
            raise ValueError("Determinant of correlation matrix is non-positive")
    except np.linalg.LinAlgError:
        raise ValueError("Failed to invert correlation matrix")

    # Log-posterior function for w
    def log_posterior(w):
        if np.any(w <= 0):
            if debug:
                print("w <= 0 detected:", w)
            return -np.inf

        # Likelihood term
        Lw = np.dot(L_ij, w)
        if np.any(Lw <= 0):
            if debug:
                print("Lw <= 0 detected:", Lw)
            return -np.inf
        lik = np.sum(np.log(Lw + 1e-10))

        # Log weights with range restriction
        logw = np.log(w)
        for j in range(n):
            logw[j] = np.clip(logw[j], min(ranges_list[j]), max(ranges_list[j]))

        # CDFs
        u = np.array([cdf_list[j](logw[j]) for j in range(n)])
        u = np.clip(u, 1e-10, 1 - 1e-10)

        # Inverse CDF
        z = norm.ppf(u)

        # Gaussian copula
        quad_term = -0.5 * np.dot(z, np.dot(P_inv - np.eye(n), z))
        log_c = quad_term - 0.5 * np.log(det_P)
        if not np.isfinite(log_c):
            if debug:
                print("Non-finite copula term:", log_c, z)
            return -np.inf

        # Marginal term
        marg = 0
        for j in range(n):
            pdf_val = max(pdf_list[j](logw[j]), 1e-8)  # Higher positivity threshold
            marg += np.log(pdf_val)

        # Jacobian term
        jac = -np.sum(logw)

        if debug:
            print(f"Likelihood: {lik:.6f}, Copula: {log_c:.6f}, Marginal: {marg:.6f}, Jacobian: {jac:.6f}")
            print(f"Total log-posterior: {lik + log_c + marg + jac:.6f}")

        return lik + log_c + marg + jac

    # Gradient of log-posterior
    def log_posterior_grad(w):
        if np.any(w <= 0):
            return np.full(n, np.inf)

        grad = np.zeros(n)
        logw = np.log(w)
        for j in range(n):
            logw[j] = np.clip(logw[j], min(ranges_list[j]), max(ranges_list[j]))
        Lw = np.dot(L_ij, w) + 1e-10
        u = np.array([cdf_list[j](logw[j]) for j in range(n)])
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = norm.ppf(u)
        phi_prime = 1 / norm.pdf(z)
        F_prime = np.array([max(pdf_list[j](logw[j]), 1e-8) for j in range(n)])

        # Likelihood gradient
        grad += np.sum(L_ij / Lw[:, np.newaxis], axis=0)

        # Copula gradient
        grad_c = np.dot(P_inv - np.eye(n), z)
        grad += (grad_c * phi_prime * F_prime) / w

        # Marginal gradient
        for k in range(n):
            try:
                d_log_p = pdf_list[k].derivative()(logw[k]) / max(pdf_list[k](logw[k]), 1e-8)
            except AttributeError:
                h = 1e-6
                log_p_plus = np.log(max(pdf_list[k](logw[k] + h), 1e-8))
                log_p_minus = np.log(max(pdf_list[k](logw[k] - h), 1e-8))
                d_log_p = (log_p_plus - log_p_minus) / (2 * h)
            grad[k] += d_log_p / w[k]

        # Jacobian gradient
        grad -= 1 / w

        if debug and not np.all(np.isfinite(grad)):
            print("Non-finite gradient:", grad)
        return grad

    if constrained:
        # Softmax transformation to enforce sum w_i = 1
        def w_from_theta(theta):
            theta = np.array(theta)
            exp_theta = np.exp(theta - np.max(theta))  # Subtract max for stability
            w = exp_theta / np.sum(exp_theta)
            w = np.clip(w, 1e-10, None)  # Ensure positivity
            return w

        def objective_theta(theta):
            w = w_from_theta(theta)
            return -log_posterior(w)

        def grad_theta(theta):
            w = w_from_theta(theta)
            grad_w = log_posterior_grad(w)
            # Softmax Jacobian
            exp_theta = np.exp(theta - np.max(theta))
            sum_exp = np.sum(exp_theta)
            dw_dtheta = np.diag(exp_theta / sum_exp) - np.outer(exp_theta, exp_theta) / sum_exp**2
            grad = -np.dot(grad_w, dw_dtheta)
            if debug and not np.all(np.isfinite(grad)):
                print("Non-finite theta gradient:", grad)
            return grad

        # Initial guess in theta space
        initial_theta = np.log(initial_w + 1e-10)
        bounds_theta = [(-50, 50)] * n  # Wide bounds for stability

        if solver == 'trust-constr':
            result = minimize(objective_theta, initial_theta, method='trust-constr', bounds=bounds_theta,
                             jac=grad_theta, options={'disp': True, 'maxiter': maxiter, 'xtol': 1e-8})
        else:
            result = minimize(objective_theta, initial_theta, method='SLSQP', bounds=bounds_theta,
                             jac=grad_theta, options={'disp': True, 'maxiter': maxiter, 'ftol': 1e-8})
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        optimal_w = w_from_theta(result.x)
    else:
        # Unconstrained optimization
        objective = lambda w: -log_posterior(w)
        grad_objective = lambda w: -log_posterior_grad(w)
        bounds = [(1e-10, None)] * n
        result = minimize(objective, initial_w, method='L-BFGS-B', bounds=bounds,
                         jac=grad_objective, options={'disp': True, 'maxiter': maxiter, 'ftol': 1e-8})
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        optimal_w = result.x

    # Normalize if unconstrained
    sum_w = np.sum(optimal_w)
    if not constrained and sum_w > 0:
        optimal_w_normalized = optimal_w / sum_w
        print(f"Unconstrained sum before normalization: {sum_w:.6f}. After normalization: {np.sum(optimal_w_normalized):.6f}")
        optimal_w = optimal_w_normalized
        sum_w = np.sum(optimal_w)

    log_post_value = log_posterior(optimal_w)
    if debug:
        print(f"Optimal weights: {optimal_w}")
        print(f"Log-posterior value: {log_post_value:.6f}")
        print(f"Sum of weights: {sum_w:.6f}")

    return optimal_w, log_post_value, sum_w
