import assets as at
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
import pandas as pd


class Portfolio:
    def __init__(self, assets: at.Assets, weights, risk_free_rate):
        self._assets = assets
        self._risk_free_rate = risk_free_rate
        if len(weights) == assets.get_number_of_assets():
            self._weights = weights
        else:
            raise Exception("Wrong number of weights")
        return

    def set_weights(self, weights):
        if len(weights) == self._assets.get_number_of_assets():
            self._weights = weights
        else:
            raise Exception("Wrong number of weights")
        return

    def get_expected_return(self):
        wT = self._weights.reshape(1, len(self._weights))  # 1 x n
        r = np.array(self._assets.get_all_assets_average_return())  # n x 1
        int1m = np.matmul(wT, r)
        return int1m

    def get_expected_volatility(self):
        if self._assets.is_covariance_matrix_stale():
            self._assets.compute_and_set_covariance_matrix()
        wT = self._weights.reshape(1, len(self._weights))  # 1 x n
        V = self._assets.get_covariance_matrix()  # n x n
        w = self._weights.reshape(len(self._weights), 1)  # n x 1
        intm1 = np.matmul(wT, V)  # (1 x n) x (n x n) = 1 x n
        intm2 = np.matmul(intm1, w)  # (1 x n) x (1 x 1) = 1 x 1
        return np.sqrt(intm2)

    def get_risk_free_rate(self):
        return self._risk_free_rate

    def get_sharpe_ratio(self):
        return (self.get_expected_return() - self.get_risk_free_rate()) / self.get_expected_volatility()

    def get_cal_m_q(self):
        return self.get_risk_free_rate(), self.get_sharpe_ratio()

    def get_assets(self):
        return self._assets


def optimal_portfolio_frontier_correct(avg_returns_vec, covariance_mat, target_ret_scalars):
    n = len(avg_returns_vec)
    P = opt.matrix(covariance_mat)
    q = opt.matrix(0.0, (n, 1))
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    a_vector = list()
    for a in avg_returns_vec:
        a_vector.append(1)
        a_vector.append(np.asscalar(a))
    A = opt.matrix(a_vector, (2, n))
    bs = list()
    for tr in target_ret_scalars:
        bs.append(opt.matrix([1, tr]))

    portfolios = [solvers.qp(P, q, G, h, A, b)['x'] for b in bs]

    weights = [[a for a in x] for x in portfolios]
    returns = [blas.dot(x, opt.matrix(avg_returns_vec)) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, P * x)) for x in portfolios]

    return weights, returns, risks


def optimal_portfolio_frontier_conditioned(avg_returns_vec, covariance, n_points_frontier=100):
    # returns is a n_assets x k_returns_observations matrix
    # covariance is a n_assets x n_assets matrix, the covariance matrix of returns matrix
    # credits to https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/
    n = len(avg_returns_vec)
    returns = np.asmatrix(avg_returns_vec)

    N = n_points_frontier
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    # S = opt.matrix(np.cov(returns))
    S = opt.matrix(covariance)
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x'] for mu in mus]

    # CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    # CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])

    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


def compute_tangency_portfolio(avg_returns_vec, covariance, risk_free_rate):
    # compute tangency pf
    inv_cov = np.linalg.inv(covariance)
    er_minus_rf_i = np.subtract(avg_returns_vec, risk_free_rate)
    ones_row = np.ones((1, len(avg_returns_vec)))
    vt_numerator = np.matmul(inv_cov, er_minus_rf_i)
    intm_1 = np.matmul(ones_row, inv_cov)
    vt_denominator = np.matmul(intm_1, er_minus_rf_i)
    tpf = np.divide(vt_numerator, vt_denominator)
    tpfr = list()
    for tpfritem in tpf:
        tpfr.append(np.asscalar(tpfritem))
    return tpfr


def compute_mgv_portfolio(avg_returns_vec, covariance):
    # compute mgv pf
    inv_cov = np.linalg.inv(covariance)
    ones_row = np.ones((1, len(avg_returns_vec)))
    ones_col = np.transpose(ones_row)
    gmv_numerator = np.matmul(inv_cov, ones_col)
    intm_2 = np.matmul(ones_row, inv_cov)
    gmv_denominator = np.matmul(intm_2, ones_col)
    return np.divide(gmv_numerator, gmv_denominator)


def optimal_portfolio_frontier_as_combination_of_tpf_and_mgvpf(avg_returns_vec, covariance, risk_free_rate,
                                                               n_points_frontier=1000):
    # compute tange and mgv pfs
    w_tang = compute_tangency_portfolio(avg_returns_vec, covariance, risk_free_rate)
    w_gmv = compute_mgv_portfolio(avg_returns_vec, covariance)

    optimal_frontier_w = list()
    optimal_frontier_r = list()
    optimal_frontier_s = list()
    for i in range(n_points_frontier):
        intm_3 = np.multiply(1.0 - i / 100.0, w_tang)
        intm_3_a = np.squeeze(np.asarray(intm_3))
        intm_4 = np.multiply(i / 100.0, w_gmv)
        intm_4_a = np.squeeze(np.asarray(intm_4))
        current_w = np.add(intm_3_a, intm_4_a)
        optimal_frontier_w.append(current_w)
        optimal_frontier_r.append(np.asscalar(np.matmul(np.transpose(current_w), avg_returns_vec)))
        optimal_frontier_s.append(np.asscalar(np.matmul(np.matmul(np.transpose(current_w), covariance), current_w)))
    return optimal_frontier_w, optimal_frontier_r, optimal_frontier_s


def get_optimal_w_s_given_target_return(target_return, risk_free_rate, covariance, avg_returns_vec):
    inv_cov = np.linalg.inv(covariance)
    er_minus_rf_i = np.subtract(avg_returns_vec, risk_free_rate)
    big_h = np.asscalar(np.matmul(np.matmul(np.transpose(er_minus_rf_i), inv_cov), er_minus_rf_i))
    optimal_w = np.matmul(np.multiply(big_h ** -1 * (target_return - risk_free_rate), inv_cov), er_minus_rf_i)
    optimal_s = np.asscalar(np.matmul(np.matmul(np.transpose(optimal_w), covariance), optimal_w))
    return optimal_w, optimal_s


def plot_pf_lines(weights_list, target_assets_name):
    # plot weights
    plt.figure()
    lines = plt.plot(weights_list)
    i = 0
    for line in lines:
        line.set_label(target_assets_name[i])
        i += 1
    plt.ylabel("weight (a different color per asset)")
    plt.xlabel("portfolio id")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=8, mode="expand", borderaxespad=0.)


def plot_pf_stacked(weights_list, target_assets_name):
    # better viz (stacked bars)
    plt.figure()
    portfolio_index = [x for x in range(len(weights_list))]
    portfolio_component_index = [x for x in range(len(weights_list[0]))]
    bar_components = list()
    for portfolio_component_idx in portfolio_component_index:
        bar_component = list()
        for portfolio_idx in portfolio_index:
            bar_component.append(abs(weights_list[portfolio_idx][portfolio_component_idx]))
        bar_components.append(bar_component)

    # build a dataframe
    a_dict = dict()
    idx = 0
    for bar_component in bar_components:
        key_name = target_assets_name[idx]
        idx += 1
        a_dict[key_name] = bar_component

    # plot
    df = pd.DataFrame(a_dict)
    df.astype(df.dtypes).plot.bar(stacked=True)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=8, mode="expand", borderaxespad=0.)

