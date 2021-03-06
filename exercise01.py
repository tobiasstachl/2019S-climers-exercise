#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
120.081 Climate and Environmental Remote Sensing (VU, 2018S) 
Exercise 1: Climate trends and variability
Example Python script to read, plot, and analyze the data
Wouter Dorigo, Matthias Forkel, and Leander Moesinger

Climate and Environmental Remote Sensing Group, Department of Geodesy and Geoinformation, Technische Universität Wien, Gusshausstraße 27-29, 1040 Vienna, Austria
 
Correspondence to: 
Matthias Forkel (matthias.forkel@geo.tuwien.ac.at): for general questions regarding the exercise
Leander Moesinger (Leander.Moesinger@geo.tuwien.ac.at): for Python-related questions

"""

# import packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import statsmodels.api as sm
from scipy import stats

import seaborn as sns
sns.set(context='paper', style='ticks')


def read_data(infile, parse_col, date_str):
    """
    Function to read data.

    Parameters
    ----------
    infile : str
        Path to data.
    parse_col : str
        Column name of time variable.
    date_str : str
        String for datetime conversion.

    returns
    -------
    data : pd.DataFrame
    """
    def dateparse(x): return pd.datetime.strptime(x, date_str)
    data = pd.read_csv(infile, parse_dates=[parse_col], date_parser=dateparse)
    data.index = data[parse_col]
    return data


def label_pval(pval):
    if pval is None:
        ValueError('No p-value given!')
    if pval < 0.001:
        label = 'p < 0.001'
    else:
        label = 'p = {:.3f}'.format(pval)

    return label


def plot_sealevel(fn_sealevel, outpath):
    """
    Visualize sealevel data and trends.

    Parameters
    ----------
    fn_sealevel : str
        Name of sealevel data file.
    outpath : str
        Name of output directory.
    """
    # create output directory if it does not yet exist
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass

    # read sealevel data
    data = read_data(infile=fn_sealevel, parse_col='Time', date_str='%Y-%m-%d')
    data = data.drop(['Time', 'Year', 'Month'], axis=1)

    # plot the data
    plt.figure(figsize=(12, 5))
    columns = ['Glob', 'NHem', 'SHem']
    plt.plot(data[columns])
    sns.despine()
    plt.title("Global sea-level anomalies", fontweight='bold')
    plt.ylabel("Sea level anomaly (m)")
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(columns, bbox_to_anchor=(1.01, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'ex1_sealevel-global-anomalies-ts.png'))

    zones = {}
    zones['north'] = ['b24N.90N', 'b64N.90N', 'b44N.64N', 'b24N.44N']
    zones['south'] = ['b90S.24S', 'b90S.64S', 'b44S.24S', 'b64S.44S']
    zones['equ'] = ['bEQU.24N', 'b24S.24N', 'b24S.EQU']
    # columns_zonal = ['b24N.90N', 'b24S.24N', 'b90S.24S', 'b64N.90N', 'b44N.64N', 'b24N.44N', 'bEQU.24N', 'b24S.EQU', 'b44S.24S', 'b64S.44S', 'b90S.64S', 'Nino34']

    for key in zones.keys():
        col = zones[key]
        plt.figure(figsize=(12, 5))
        plt.plot(data[col])
        sns.despine()
        plt.title("Latitudinal band sea-level anomalies", fontweight='bold')
        plt.ylabel("Sea level anomaly (m)")
        plt.ylim(-0.15, 0.15)
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.legend(col, bbox_to_anchor=(1.01, 0.5), loc="center left")
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, 'ex1_sealevel-{}-anomalies-ts.png'. format(key)))

    # aggregate to annual data
    data_annual = data.resample('A').mean()

    for col in data_annual.columns:
        print(col)

        # 1) compute trend based on ordinary least square regression
        y = data_annual[col]
        x = list(range(1, len(data_annual) + 1))
        x = np.reshape(x, (-1, 1))
        x = sm.add_constant(x)  # to add intercept
        trd = sm.OLS(y, x).fit()  # fit regression
        data_annual["{}_trd_ols".format(col)] = trd.predict(x)  # compute trend line
        print(trd.summary())

        # 2) compute trend based on Theil-Sen slope
        y = data_annual[col]
        alpha = 0.95
        theilsen_results = theilsen(y, alpha=alpha) # p < 0.05
        x = data_annual.index.to_julian_date() # list(range(1, len(data_annual) + 1))
        data_annual['{}_trdmed'.format(col)] = theilsen_results['median_interc'] + theilsen_results['median_slope'] * x
        data_annual['{}_trdlow'.format(col)] = theilsen_results['lower_CI_interc'] + theilsen_results['lower_CI_slope'] * x
        data_annual['{}_trdupp'.format(col)] = theilsen_results['upper_CI_interc'] + theilsen_results['upper_CI_slope'] * x

        # compare trends
        plt.figure()
        data_annual[col].plot(color="black", label='Sea-level anomaly')

        label_OLS = 'OLS ($slope={:.3f}, {}$)'.format(trd.params['x1'], label_pval(trd.pvalues['x1']))
        data_annual["{}_trd_ols".format(col)].plot(color="red", label=label_OLS)
        data_annual['{}_trdmed'.format(col)].plot(color="blue", label='Theil-Sen median') # ($slope={:.3f}, p={:.2f}$)'.format(theilsen_results['median_slope'], 1-alpha))
        data_annual['{}_trdlow'.format(col)].plot(
            color="blue", linestyle='dashed', label='Lower CI')
        data_annual['{}_trdupp'.format(col)].plot(
            color="blue", linestyle='dashed', label='Upper CI')

        # figure aesthetics
        sns.despine()
        plt.title("Sea-level anomalies and trends: {}".format(col), fontweight='bold')
        plt.ylabel("Sea level anomaly (m)")

        plt.ylim(-0.05, 0.08)
        if col == 'Nino34':
            plt.ylim(-0.08, 0.2)

        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(outpath, 'ex1_sealevel_trend_{}.png'.format(col)))
        plt.close()


def plot_surf_temp(fn_temp, outpath):
    """
    Visualize global surface temperature data and trends.


    Parameters
    ----------
    fn_temp : str
        Name of temperature data file.
    outpath : str
        Name of output directory.
    """
    # create output directory if it does not yet exist
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass

    # read temperature data
    data = read_data(infile=fn_temp, parse_col='Year', date_str='%Y')
    data = data.drop(['Year'], axis=1)

    # plot global data
    plt.figure(figsize=(12, 5))
    columns = ['Glob', 'NHem', 'SHem']

    plt.plot(data[columns])
    sns.despine()
    plt.title("Global surface temperature anomalies", fontweight='bold')
    plt.ylabel("Temperature anomaly (°C)")
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(columns, bbox_to_anchor=(1.01, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'ex1_surftemp-global-anomalies-ts.png'))

    zones = {}
    zones['north'] = ['24N-90N', '64N-90N', '44N-64N', '24N-44N']
    zones['south'] = ['90S-24S', '90S-64S', '44S-24S', '64S-44S']
    zones['equ'] = ['EQU-24N', '24S-24N', '24S-EQU']

    for key in zones.keys():
        col = zones[key]
        plt.figure(figsize=(12, 5))
        plt.plot(data[col])
        sns.despine()
        plt.title("Latitudinal band surface temperature anomalies", fontweight='bold')
        plt.ylabel("Temperature anomaly (°C)")
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.legend(col, bbox_to_anchor=(1.01, 0.5), loc="center left")
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, 'ex1_surftemp-{}-anomalies-ts.png'.format(key)))

    # aggregate to annual data
    data_annual = data.resample('A').mean()

    for col in data_annual.columns:

        # 1) compute trend based on ordinary least square regression
        y = data_annual[col]
        x = list(range(1, len(data_annual) + 1))
        x = np.reshape(x, (-1, 1))
        x = sm.add_constant(x)  # to add intercept
        trd = sm.OLS(y, x).fit()  # fit regression
        data_annual["{}_trd_ols".format(col)] = trd.predict(x)  # compute trend line
        # print(trd.summary())

        # compute trend based on Theil-Sen slope
        y = data_annual[col]
        theilsen_results = theilsen(y, alpha=0.95)
        x = data_annual.index.to_julian_date() # list(range(1, len(data_annual) + 1))
        data_annual['{}_trdmed'.format(col)] = theilsen_results['median_interc'] + theilsen_results['median_slope'] * x
        data_annual['{}_trdlow'.format(col)] = theilsen_results['lower_CI_interc'] + theilsen_results['lower_CI_slope'] * x
        data_annual['{}_trdupp'.format(col)] = theilsen_results['upper_CI_interc'] + theilsen_results['upper_CI_slope'] * x

        # compare trends
        plt.figure()
        data_annual[col].plot(color="black", label='Temperature anomaly (°C)')

        label = 'OLS ($slope={:.3f}, {}$)'.format(trd.params['x1'], label_pval(trd.pvalues['x1']))
        data_annual["{}_trd_ols".format(col)].plot(color="red", label=label)
        data_annual['{}_trdmed'.format(col)].plot(color="blue", label='Theil-Sen median')
        data_annual['{}_trdlow'.format(col)].plot(
            color="blue", linestyle='dashed', label='Lower CI')
        data_annual['{}_trdupp'.format(col)].plot(
            color="blue", linestyle='dashed', label='Upper CI')

        # figure aesthetics
        sns.despine()
        plt.ylim(-2.5, 3)
        plt.title("Global surface temperature anomalies and trends: {}".format(col), fontweight='bold')
        plt.ylabel("Temperature anomaly (°C)")
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, 'ex1_surftemp_trend_{}.png'.format(col)))
        plt.close()


def detrend(s):
    y = s.values
    x = list(range(1, len(s) + 1))
    x = np.reshape(x, (-1, 1))
    x = sm.add_constant(x)  # to add intercept
    trd = sm.OLS(y, x).fit()  # fit regression
    trend = trd.predict(x)  # compute trend line
    return pd.Series(data=(y-trend), index=s.index)

def theilsen(s, alpha=0.95):
    """
    Function to compute Theil-Sen slopes.
    :param s: pandas.Series
    :param alpha: Confidence degree between 0 and 1. Default is 95% confidence.
    :return: trd: dictionary containing intercept and slope of the median, upper CI and lower CI respectively.

    """
    x = s.index.to_julian_date()
    x = np.reshape(x, (-1, 1))
    trd = stats.theilslopes(s, x, alpha=alpha)
    interc_low = np.median(s) - np.median(x) * trd[2]
    interc_upp = np.median(s) - np.median(x) * trd[3]
    trd = pd.Series({
        'median_slope': trd[0],
        'median_interc': trd[1],
        'lower_CI_slope': trd[2],
        'lower_CI_interc': interc_low,
        'upper_CI_slope': trd[3],
        'upper_CI_interc': interc_upp
        })
    return trd


def plots_taskb(fn_temp, outpath):
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass

    # read sealevel data
    data = read_data(infile=fn_temp, parse_col='Year', date_str='%Y')
    print(data.columns)

    # select zone
    for zone in ['Glob', 'NHem', 'SHem', '24N-90N', '24S-24N', '90S-24S',
       '64N-90N', '44N-64N', '24N-44N', 'EQU-24N', '24S-EQU', '44S-24S',
       '64S-44S', '90S-64S']:
        #col = 'Glob'

        # plot temperature anomalies from 1970 onwards
        plt.figure(figsize=(10,4))
        plt.plot(data[zone]['01-01-1970':], linestyle='--')

        # compute trends in loop and plot
        for period in [(1990, 2005), (1998, 2014), (1998, 2012), (2003, 2018)]:
            start, end = period
            end = 2018 if end > 2018 else end
            mask = (data['Year'] >= str(start)) & (data['Year'] <= str(end))
            data_temp_masked = data.loc[mask]
            print(data_temp_masked)

            y = data_temp_masked[zone]

            x = list(range(1, len(data_temp_masked) + 1))
            x = np.reshape(x, (-1, 1))
            x = sm.add_constant(x)  # to add intercept
            trd = sm.OLS(y, x).fit()  # fit regression
            data_temp_masked["{}_trd_ols".format(zone)] = trd.predict(x)  # compute trend line
            print(trd.summary())
            print(trd.params['x1'])

            # plot trends
            (data_temp_masked["{}_trd_ols".format(zone)]
             .plot(label=r'OLS for {}-{} ($slope={:.3f}, {}$)'.format(start,
                                                                      end,
                                                                      trd.params['x1'],
                                                                      label_pval(trd.pvalues['x1'])),
             linewidth=2, linestyle='-'))

        sns.despine()
        plt.title("{} Surface temperature anomalies (Base period 1951-1980)".format(zone))
        plt.ylabel("Temperature anomaly (°C)")
        plt.grid(color='grey', linestyle='--', linewidth=0.5, axis='both')
        plt.tight_layout()
        plt.legend()

        #plt.show()
        plt.savefig(os.path.join(outpath, 'ex1-{}-warming-pause.png'.format(zone)), dpi=150)


def task_c(fn_sealevel, fn_temp, fn_soi, outpath):
    """
    Task C

    Parameters
    ----------
    fn_sealevel : str
        Name of sealevel data file.
    fn_temp : str
        Name of temperature data file.
    fn_soi : str
        Name of SOI data file.
    outpath : str
        Name of output directory.
    """
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass

    # read data
    sealevel = read_data(infile=fn_sealevel, parse_col='Time', date_str='%Y-%m-%d')
    sealevel = sealevel.drop(['Time', 'Year', 'Month'], axis=1)

    temp = read_data(infile=fn_temp, parse_col='Year', date_str='%Y')
    temp = temp.drop(['Year'], axis=1)

    soi = read_data(infile=fn_soi, parse_col='Date', date_str='%Y%m')
    soi = soi['Value'].rename('SOI')
    soi = pd.DataFrame(soi)

    # compute trends for sea-level data
    # ----------------------------------

    # aggregate to annual data centered at the last day of the year
    sealevel_annual = sealevel.resample('A').mean() # was M
    temp_annual = temp.resample('A').mean() # was AS
    soi_annual = soi.resample('A').mean() # was M


    # rename matching zons with slightly different column names to match amongst each other
    sealevel_annual_nino34 = sealevel_annual['Nino34']
    sealevel_annual = sealevel_annual.drop(['Nino34'], axis=1)
    sealevel_annual.columns = temp_annual.columns.values
    sealevel_annual['Nino34'] = sealevel_annual_nino34

    # ------------------------------------------------------------------------------------
    # Experiment A: annual data within common date range of all 3 data sets (1993-2015)
    # ------------------------------------------------------------------------------------
    # 1) cut two-fold combinations to same range
    sealevel_annual_1993_2015 = sealevel_annual['1993-01-01':'2015-12-31']
    soi_annual_1993_2015 = soi_annual['1993-01-01':'2015-12-31']

    soi_annual_1951_2018 = soi_annual['1951-01-01':'2018-12-31']
    temp_annual_1951_2018 = temp_annual['1951-01-01':'2018-12-31']

    # 2) de-trend sea level and temperature anomalies
    sealevel_annual_1993_2015_detrended = sealevel_annual_1993_2015.apply(detrend)
    temp_annual_1951_2018_detrended = temp_annual_1951_2018.apply(detrend)
    # ------------------------------------------------------------------------------------
    # 4a) compute correlations of sea level with SOI
    # ------------------------------------------------------------------------------------
    merged_soi_sealevel = pd.concat([soi_annual_1993_2015, sealevel_annual_1993_2015_detrended], axis=1)
    corr_soi_sealevel = merged_soi_sealevel.corr()

    # create triangular mask for heatmap
    mask = np.zeros_like(corr_soi_sealevel)
    mask[np.triu_indices_from(mask)] = True

    # plot heatmap of pairwise correlations
    f, ax = plt.subplots(figsize=(8, 8))

    # define correct cbar height and pass to sns.heatmap function
    cbar_kws = {"fraction": 0.046, "pad": 0.04}
    sns.heatmap(corr_soi_sealevel, mask=mask, cmap='coolwarm_r', square=True, vmin=-1, vmax=1, annot=True, cbar_kws=cbar_kws, ax=ax)
    plt.title("Correlation between annual SOI and annual average sea level over different zones (Period 1993-2015)")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'ex3_heatmap_SOI_sealevel_detrended.png'), dpi=150)
    # ------------------------------------------------------------------------------------
    # 4b) compute correlations of surface temperature with SOI
    # ------------------------------------------------------------------------------------
    merged_soi_surftemp = pd.concat([soi_annual_1951_2018, temp_annual_1951_2018_detrended], axis=1)
    corr_soi_surftemp = merged_soi_surftemp.corr()

    # create triangular mask for heatmap
    mask = np.zeros_like(corr_soi_surftemp)
    mask[np.triu_indices_from(mask)] = True

    # plot heatmap of pairwise correlations
    f, ax = plt.subplots(figsize=(8, 8))
    cbar_kws = {"fraction": 0.046, "pad": 0.04}
    sns.heatmap(corr_soi_surftemp, mask=mask, cmap='coolwarm_r', square=True, vmin=-1, vmax=1, annot=True,
                cbar_kws=cbar_kws, ax=ax)
    plt.title("Correlation between annual SOI and annual average surface temperature over different zones (Period 1951-2018)")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'ex3_heatmap_SOI_surftemp_detrended.png'), dpi=150)

    #plt.show()
    pass


if __name__ == '__main__':
    # Gets the directory where this python script is located,
    # which is assumed to be in "CLIMERS_ex01" with subdirectory "data"
    inpath = os.path.dirname(os.path.realpath(__file__))

    # definition of file names
    outpath = os.path.join(inpath, 'results')
    fn_sealevel = os.path.join(inpath, 'data',
                               'ESACCI-SEALEVEL-L4-MSLA-MERGED-1993-2015-fv02',
                               'ESACCI-SEALEVEL-L4-MSLA-MERGED-1993-2015-fv02_Zonal.csv')
    fn_temp = os.path.join(inpath, 'data', 'NASA-GISSTEMP',
                           'nasa_giss_dTs+dSST_ZonAnn.csv')
    fn_soi = os.path.join(inpath, 'data', 'NOAA-NCDC-SOI',
                          'noaa_ncdc_soi_data.csv')

    #---------------------------------------------------------------------------
    # Analysis A.1.: Trends in sea level anomalies
    # --------------------------------------------------------------------------
    # plot_sealevel(fn_sealevel, os.path.join(outpath, 'sealevel'))

    # --------------------------------------------------------------------------
    # Analysis A.2.: Trends in surface temperature anomalies
    # --------------------------------------------------------------------------
    # plot_sst(fn_temp, os.path.join(outpath, 'temperature'))

    # --------------------------------------------------------------------------
    # Analysis B: Global warming hiatus 1998-2012/2014
    # --------------------------------------------------------------------------
    # plots_taskb(fn_temp, os.path.join(outpath, 'taskb-warming-hiatus'))

    # --------------------------------------------------------------------------
    # Analysis C: El Nino and climate variability
    # --------------------------------------------------------------------------
    # What to expect:
    #    1) significant correlation between SOI time series
    #       and Surface Temperature/Sea level anomalies over eastern pacific
    #    2) similar trend in SOI, SST and Sea level over eastern pacific
    #    3) at the global scale?
    # --------------------------------------------------------------------------
    task_c(fn_sealevel, fn_temp, fn_soi, os.path.join(outpath, 'taskc-SOI-corr'))