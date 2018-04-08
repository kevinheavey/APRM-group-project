"""A collection of functions written for the APRM group project"""

import numpy as np
import empyrical as ep
import pyfolio_fork_aprm as pf
from functools import partial
import pandas as pd
from collections import namedtuple
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

ACTIVE = ['JFUAX', 'JLCAX', 'JDEAX', 'OEIAX', 'JFTAX']
PASSIVE = ['BWIIX', 'BSPIX', 'MAIIX']



def drawdown_alt(returns_series):
    price_series = returns_series.add(1).cumprod()
    fund_nrow = price_series.shape[0]
    Pdrop = np.zeros(fund_nrow, dtype=bool)
    for i in range(1, fund_nrow):
        if price_series.iloc[i] < price_series[i - 1]:
            Pdrop[i] = True
    Min = 0
    i = 0
    j = 0
    while (i < fund_nrow - 1):
        j = i + 1
        if Pdrop[j]:
            while (j < fund_nrow - 1):
                if Pdrop[j + 1]:
                    j = j + 1
                else:
                    if price_series.iloc[j] / price_series.iloc[i] - 1 < Min:
                        Min = price_series.iloc[j] / price_series.iloc[i] - 1
                    break
            if j == fund_nrow - 1:
                if price_series.iloc[j] / price_series.iloc[i] - 1 < Min:
                    Min = price_series.iloc[j] / price_series.iloc[i] - 1
        i = j
    return np.absolute(Min)


def adjusted_pow(series, order):
    if order == 0:
        return series.where(series == 0, 1)
    else:
        return series.pow(order)


def lpm(excess_returns_series, target_rate, order):
    """Computes the lower partial moment of a series of returns. 
    
    Note: this is already an unbiased estimator if our target_rate is not an unknown
    population parameter which we have estimated (e.g. the mean). 
    The sample raw moment is an unbiased estimator of the raw moment. 
    See http://mathworld.wolfram.com/SampleRawMoment.html
    For unbiased estimators of central moments, see http://mathworld.wolfram.com/SampleCentralMoment.html"""
    return (excess_returns_series
            .sub(target_rate)
            .clip_upper(0)
            .pipe(adjusted_pow, order=order)
            .mean())


def shortfall_risk(excess_returns_series):
    return lpm(excess_returns_series, 0, 0)


def expected_shortfall(excess_returns_series):
    return -excess_returns_series[excess_returns_series < 0].mean()


def lpm_0_2(excess_returns_series):
    return lpm(excess_returns_series, target_rate=0, order=2)


def root_lpm_0_2_annualised(excess_returns_series):
    return np.sqrt(252 * lpm_0_2(excess_returns_series))


def information_ratio_annualised(returns_series, index_returns_series):
    active_returns = returns_series - index_returns_series
    return ep.sharpe_ratio(active_returns)


def treynor_ratio_annualised(excess_returns, index_excess_returns):
    _beta = ep.beta(excess_returns, index_excess_returns)
    return 252 * excess_returns.mean() / _beta


def sterling_ratio_annualised(excess_returns_series):
    mean_drawdown = excess_returns_series.resample('Y').apply(drawdown_alt).mean()
    mean_return = excess_returns_series.mean()
    return 252 * mean_return / mean_drawdown


def burke_ratio_annualised(excess_returns_series):
    root_mean_square_drawdown = np.sqrt(excess_returns_series.resample('Y').apply(drawdown_alt).pow(2).mean())
    mean_return = excess_returns_series.mean()
    return 252 * mean_return / root_mean_square_drawdown


def rops_annualised(excess_returns_series):
    return 252 * excess_returns_series.mean() / shortfall_risk(excess_returns_series)


def kappa1(excess_returns_series):
    return excess_returns_series.mean() / lpm(excess_returns_series, 0, 1)


def kappa3(excess_returns_series):
    return excess_returns_series.mean() / np.cbrt(lpm(excess_returns_series, 0, 3))


def rovar(excess_returns_series):
    return excess_returns_series.mean() / np.abs(ep.value_at_risk(excess_returns_series))


def corrected_semivariance(excess_returns_series):
    n = len(excess_returns_series)
    correction_factor = n / (n - 1)
    return lpm(excess_returns_series, excess_returns_series.mean(), 2) * correction_factor


def annualised_semi_std(excess_returns_series):
    return corrected_semivariance(excess_returns_series) * np.sqrt(252)


def get_perf_df(returns_series, risk_free_returns_series, factor_returns_series):
    drawdown = drawdown_alt(returns_series)
    excess_returns = returns_series - risk_free_returns_series
    factor_excess_returns = factor_returns_series - risk_free_returns_series
    perf_df_raw = (pf.plotting.show_perf_stats(excess_returns, factor_returns=factor_excess_returns,
                                               suppress_print=True, backtest_label_replacement=returns_series.name,
                                               return_df=True, suppress_display=True)
                   .drop('Calmar ratio')  # not worth the hassle
                   .rename({'Alpha': 'Alpha (annualised)',
                            'Max drawdown': 'Drawdown',
                            'Sharpe ratio': 'Sharpe ratio (annualised)',
                            'Omega ratio': 'Omega ratio (annualised)',
                            'Sortino ratio': 'Sortino ratio (annualised)',
                            'Daily value at risk': 'Value at risk',
                            'Annual volatility': 'Volatility (annualised)'}))
    perf_df_raw.loc['Drawdown'] = drawdown
    return perf_df_raw


EXTRA_PERCENT_ROWS = ['Annual downside risk (lpm_02)',
                      'Expected shortfall',
                      'Drawdown',
                      'Annual semi-standard deviation',
                      'Annual volatility',
                      'Shortfall risk',
                      'Return on prob. of shortfall (annualised)',
                      'Return on value at risk']


def get_extra_perf_stats(returns_series, risk_free_returns_series, index_returns_series):
    index_excess_returns_series = index_returns_series - risk_free_returns_series
    excess_returns_series = returns_series - risk_free_returns_series
    extra_stats = {'Shortfall risk': shortfall_risk,
                   'Expected shortfall': expected_shortfall,
                   'root(lpm(0,2)) (annualised)': root_lpm_0_2_annualised,
                   'Information ratio (annualised)': partial(information_ratio_annualised,
                                                             index_returns_series=index_excess_returns_series),
                   'Treynor ratio (annualised)': partial(treynor_ratio_annualised,
                                                         index_excess_returns=index_excess_returns_series),
                   'Sterling ratio (annualised)': sterling_ratio_annualised,
                   'Burke ratio (annualised)': burke_ratio_annualised,
                   'Return on prob. of shortfall (annualised)': rops_annualised,
                   'Kappa 1': kappa1,
                   'Kappa 3': kappa3,
                   'Return on value at risk': rovar}
    stats_dict = {key: val(excess_returns_series) for key, val in extra_stats.items()}
    return pd.Series(stats_dict).to_frame(returns_series.name)


def get_full_perf_stats(returns_series, risk_free_returns_series, index_returns_series):
    returns_series = returns_series.dropna()  # one of the columns has some NaNs
    full_perf_df = (get_perf_df(returns_series,
                                risk_free_returns_series,
                                index_returns_series)
                    .append(get_extra_perf_stats(returns_series,
                                                 risk_free_returns_series,
                                                 index_returns_series))
                    )

    for stat, value in full_perf_df[returns_series.name].iteritems():
        if stat in EXTRA_PERCENT_ROWS:
            new_val = str(np.round(value * 100, 1)) + '%'
            full_perf_df.loc[stat, returns_series.name] = new_val

    return full_perf_df


def get_stats_from_fund_obj(fund, returns_df, risk_free_returns):
    returns_series = returns_df[fund.name].dropna()
    risk_free_returns = pd.concat([risk_free_returns,
                                   returns_df[fund.name]], axis=1).dropna()['RF']
    index_returns = returns_df[fund.index_benchmark]
    return get_full_perf_stats(returns_series, risk_free_returns, index_returns)


def get_full_perf_stats_df(returns_df, funds_list, risk_free_returns_series):
    _list = [get_stats_from_fund_obj(fund, returns_df, risk_free_returns_series)
             for fund in funds_list]
    return mark_active_vs_passive_cols(pd.concat(_list, axis=1))


def get_returns_df():
    prices = (pd.read_excel('APRM_total_returns.xlsx', skiprows=[0, 1, 2, 4, 5], index_col=0)
              .rename(columns=lambda col: col.replace(' US Equity', '').replace(' Index', ''))
              )
    returns_df = prices.pct_change().tail(-1)
    returns_df.index = returns_df.index.tz_localize('UTC')
    return returns_df


def get_funds_list():
    Fund = namedtuple('Fund', 'name passive_alternative index_benchmark')
    jfuax = Fund('JFUAX', 'BWIIX', 'MXWO')
    jlcax = Fund('JLCAX', 'BSPIX', 'SPX')
    jdeax = Fund('JDEAX', 'BSPIX', 'SPX')
    oeiax = Fund('OEIAX', 'MAIIX', 'MXEA')
    jftax = Fund('JFTAX', 'MAIIX', 'MXEA')
    active_funds = [jfuax, jlcax, jdeax, oeiax, jftax]

    bwiix = Fund('BWIIX', None, 'MXWO')
    bspix = Fund('BSPIX', None, 'SPX')
    maiix = Fund('MAIIX', None, 'MXEA')
    passive_funds = [bwiix, bspix, maiix]

    funds = active_funds + passive_funds
    return funds


def get_funds_df(funds_list):
    sub_list = funds_list[:5]  # just the active ones
    return pd.DataFrame(sub_list)


def make_tables(full_perf_stats_df, funds_df, fama_french_df):
    float_format = '%0.2f'
    with pd.ExcelWriter('tables.xlsx') as w:
        full_perf_stats_df.to_excel(w, 'perf_stats', float_format=float_format)
        funds_df.to_excel(w, 'fund_matches')
        fama_french_df.to_excel(w, 'fama_french', float_format=float_format)




def get_ff_factor_data(returns_df):
    ff = ep.utils.load_portfolio_risk_factors(start=returns_df.index[0],
                                              end=returns_df.index[-1])
    return ff

def get_risk_free_returns_series(returns_df):
    ff = get_ff_factor_data(returns_df)
    # forward fill risk-free returns for a few days where they're missing
    return returns_df.join(ff, how='left')['RF'].ffill()


def fama_french_regression(returns_series, index_returns_series, extra_factors_df):
    combined_df = returns_series.to_frame().join([extra_factors_df, index_returns_series], how='inner').dropna()
    excess_returns_name = returns_series.name
    index_excess_returns_name = 'Index'
    combined_df.loc[:, excess_returns_name] = combined_df[returns_series.name] - combined_df['RF']
    combined_df.loc[:, index_excess_returns_name] = combined_df[index_returns_series.name] - combined_df['RF']
    formula = '{} ~ 1 + {} + SMB + HML + Mom'.format(excess_returns_name, index_excess_returns_name)

    return smf.ols(formula, combined_df).fit(cov_type='HAC', cov_kwds={'maxlags': 12, 'use_correction': True})

def fama_french_from_fund_obj(fund, returns_df, extra_factors_df):
    returns_series = returns_df[fund.name]
    index_returns_series = returns_df[fund.index_benchmark]
    return fama_french_regression(returns_series, index_returns_series, extra_factors_df)

def get_extra_factors_df(returns_df):
    return ep.utils.load_portfolio_risk_factors(start=returns_df.index[0], end=returns_df.index[-1])

def summary(fit_list):
    return summary_col(fit_list,
                       float_format='%0.4f',
                       info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                                  'R2':lambda x: "{:.2f}".format(x.rsquared)},stars=True
                      ).tables[0]

def mark_active_vs_passive_cols(df):
    df.columns.name = 'Fund'
    df = df.T.reset_index()
    df.loc[:, 'x'] = ['Active' if x in ACTIVE else 'Passive' for x in df['Fund']]
    df = df.set_index(['x', 'Fund']).T
    df.columns.names = [None, None]
    return df


def get_fama_french_df(funds_list, returns_df):
    extra_factors = get_extra_factors_df(returns_df)
    fit_list = [fama_french_from_fund_obj(fund, returns_df, extra_factors)
               for fund in funds_list]
    return mark_active_vs_passive_cols(summary(fit_list))



def main():
    returns_df = get_returns_df()
    risk_free_returns = get_risk_free_returns_series(returns_df)
    funds_list = get_funds_list()
    funds_df = get_funds_df(funds_list)
    full_perf_stats_df = get_full_perf_stats_df(returns_df, funds_list, risk_free_returns)
    fama_french_df = get_fama_french_df(funds_list, returns_df)
    make_tables(full_perf_stats_df, funds_df, fama_french_df)
