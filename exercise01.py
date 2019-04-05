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
    print(data.columns)

    # plot the data
    plt.figure(figsize=(12,5))
    columns = ['Glob', 'NHem', 'SHem']
    plt.plot(data[columns])
    sns.despine()
    plt.title("Global sea-level anomalies")
    plt.ylabel("Sea level anomaly (m)")
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(columns, bbox_to_anchor=(1.01, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'ex1_sealevel-global-anomalies-ts.png'))

    # plot the data
    plt.figure(figsize=(12,5))
    columns_zonal = ['b24N.90N', 'b24S.24N', 'b90S.24S', 'b64N.90N', 'b44N.64N', 'b24N.44N',
                     'bEQU.24N', 'b24S.EQU','b44S.24S', 'b64S.44S', 'b90S.64S', 'Nino34']
    plt.plot(data[columns_zonal])
    sns.despine()
    plt.title("Latitudinal band sea-level anomalies")
    plt.ylabel("Sea level anomaly (m)")
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(columns_zonal, bbox_to_anchor=(1.01, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'ex1_sealevel-zonal-anomalies-ts.png'))

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
        theilsen_results = theilsen(y, alpha=0.95) # p < 0.05
        x = data_annual.index.to_julian_date() # list(range(1, len(data_annual) + 1))
        data_annual['{}_trdmed'.format(col)] = theilsen_results['median_interc'] + theilsen_results['median_slope'] * x
        data_annual['{}_trdlow'.format(col)] = theilsen_results['lower_CI_interc'] + theilsen_results['lower_CI_slope'] * x
        data_annual['{}_trdupp'.format(col)] = theilsen_results['upper_CI_interc'] + theilsen_results['upper_CI_slope'] * x

        # compare trends
        plt.figure()
        data_annual[col].plot(color="black", label='Sea-level anomaly')
        data_annual["{}_trd_ols".format(col)].plot(color="red",
                                                   label=r'OLS ($r^2={:.2f}, p={:.3f}$)'.format(trd.rsquared,
                                                                                                trd.pvalues['x1']))
        data_annual['{}_trdmed'.format(col)].plot(color="blue", label='Theilsen ()')
        data_annual['{}_trdlow'.format(col)].plot(
            color="blue", linestyle='dashed', label='Lower CI')
        data_annual['{}_trdupp'.format(col)].plot(
            color="blue", linestyle='dashed', label='Upper CI')

        # figure aesthetics
        sns.despine()
        plt.title("Sea-level anomalies and trends: {}".format(col))
        plt.ylabel("Sea level anomaly (m)")

        plt.ylim(-0.05, 0.08)
        if col == 'Nino34':
            plt.ylim(-0.08, 0.2)

        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(outpath, 'ex1_sealevel_trend_{}.png'.format(col)))
        plt.close()

def plot_sst(fn_temp, outpath):
    """
    Visualize sealevel data and trends.

    Parameters
    ----------
    fn_temp : str
        Name of SST data file.
    outpath : str
        Name of output directory.
    """
    # create output directory if it does not yet exist
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass

    # read sealevel data
    data = read_data(infile=fn_temp, parse_col='Year', date_str='%Y')
    data = data.drop(['Year'], axis=1)
    print(data.columns)

    # plot the data
    plt.figure(figsize=(12,5))
    columns = ['Glob', 'NHem', 'SHem']
    plt.plot(data[columns])
    sns.despine()
    plt.title("Global SST anomalies")
    plt.ylabel("SST anomaly (°C)")
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(columns, bbox_to_anchor=(1.01, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'ex1_sst-global-anomalies-ts.png'))

    # plot the data
    plt.figure(figsize=(12,5))
    columns_zonal = ['24N-90N', '24S-24N', '90S-24S', '64N-90N','44N-64N',
                     '24N-44N', 'EQU-24N', '24S-EQU', '44S-24S', '64S-44S','90S-64S']
    plt.plot(data[columns_zonal])
    sns.despine()
    plt.title("Latitudinal band SST anomalies")
    plt.ylabel("SST anomaly (°C)")
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(columns_zonal, bbox_to_anchor=(1.01, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'ex1_sst-zonal-anomalies-ts.png'))


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
        print(trd.summary())

        # compute trend based on Theil-Sen slope
        y = data_annual[col]
        theilsen_results = theilsen(y, alpha=0.95)
        x = data_annual.index.to_julian_date() # list(range(1, len(data_annual) + 1))
        data_annual['{}_trdmed'.format(col)] = theilsen_results['median_interc'] + theilsen_results['median_slope'] * x
        data_annual['{}_trdlow'.format(col)] = theilsen_results['lower_CI_interc'] + theilsen_results['lower_CI_slope'] * x
        data_annual['{}_trdupp'.format(col)] = theilsen_results['upper_CI_interc'] + theilsen_results['upper_CI_slope'] * x

        # compare trends
        plt.figure()
        data_annual[col].plot(color="black", label='SST anomaly')
        data_annual["{}_trd_ols".format(col)].plot(color="red",
                                                   label=r'OLS ($r^2={:.2f}, p={:.3f}$)'.format(trd.rsquared,
                                                                                                trd.pvalues['x1']))
        data_annual['{}_trdmed'.format(col)].plot(color="blue", label='Theilsen ()')
        data_annual['{}_trdlow'.format(col)].plot(
            color="blue", linestyle='dashed', label='Lower CI')
        data_annual['{}_trdupp'.format(col)].plot(
            color="blue", linestyle='dashed', label='Upper CI')

        # figure aesthetics
        sns.despine()
        plt.ylim(-2.5, 3)
        plt.title("SST anomalies and trends: {}".format(col))
        plt.ylabel("SST anomaly (°C)")
        plt.grid(color='grey', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, 'ex1_sst_trend_{}.png'.format(col)))
        plt.close()


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


def trend_analysis(fn_sealevel, fn_temp, fn_soi, outpath):
    """
    PART 1: Global trends in sea-level and temperature: Here starts your own analysis!

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
    
    # read data
    sealevel = read_data(infile=fn_sealevel, parse_col='Time', date_str='%Y-%m-%d')
    temp = read_data(infile=fn_temp, parse_col='Year', date_str='%Y')
    soi = read_data(infile=fn_soi, parse_col='Date', date_str='%Y%m')

    # compute trends for sea-level data
    # ----------------------------------

    # aggregate to annual data
    sealevel_annual = sealevel.resample('A').mean()

    # compute Theil-Sen slopes for all regions
    trd = sealevel_annual.apply(theilsen, alpha=0.95)
    trd.to_csv(os.path.join(outpath,
                            'ESACCI-SEALEVEL-L4-MSLA-MERGED-1993-2015-fv02_trends.csv'))

    # insert here the code for your own analysis ...


if __name__ == '__main__':
    """
    run the analyses for exercise 1
    """

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


    # run the individual analyses

    # Analysis 1a: Trends in sea level anomalies
    plot_sealevel(fn_sealevel, os.path.join(outpath, 'sealevel'))

    # Analysis 1b: Trends in surface temperature anomalies
    plot_sst(fn_temp, os.path.join(outpath, 'temperature'))

    #trend_analysis(fn_sealevel, fn_temp, fn_soi, outpath)

    # insert here the code for your own analysis ...

    #plt.show()







