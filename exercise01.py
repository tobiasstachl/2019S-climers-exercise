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
import statsmodels.api as sm
from scipy import stats


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
    print(data.describe())

    # plot the data
    plt.figure()
    columns = ['Glob', 'NHem', 'SHem']
    plt.plot(data[columns])
    plt.title("Global sea-level anomalies")
    plt.ylabel("Sea level anomaly (m)")
    plt.grid(True)
    plt.legend(columns)
    plt.savefig(os.path.join(outpath, 'ex1_sealevel-ts.png'))

    # aggregate to annual data
    data_annual = data.resample('A').mean()

    # compute trend based on ordinary least square regression
    y = data_annual["Glob"]
    x = list(range(1, len(data_annual) + 1))
    x = np.reshape(x, (-1, 1))
    x = sm.add_constant(x)  # to add intercept
    trd = sm.OLS(y, x).fit()  # fit regression
    data_annual["Glob_trd_ols"] = trd.predict(x)  # compute trend line
    print(trd.summary())

    # compute trend based on Theil-Sen slope
    y = data_annual["Glob"]
    theilsen_results = theilsen(y, alpha=0.95)
    x = data_annual.index.to_julian_date() # list(range(1, len(data_annual) + 1))
    data_annual['Glob_trdmed'] = theilsen_results['median_interc'] + theilsen_results['median_slope'] * x
    data_annual['Glob_trdlow'] = theilsen_results['lower_CI_interc'] + theilsen_results['lower_CI_slope'] * x
    data_annual['Glob_trdupp'] = theilsen_results['upper_CI_interc'] + theilsen_results['upper_CI_slope'] * x

    # compare trends
    plt.figure()
    data_annual.Glob.plot(color="black", label='Sea-level anomaly')
    data_annual.Glob_trd_ols.plot(color="red", label='OLS trend')
    data_annual.Glob_trdmed.plot(color="blue", label='Theil-Slope median trend')
    data_annual.Glob_trdlow.plot(
        color="blue", linestyle='dashed', label='Confidence interval')
    data_annual.Glob_trdupp.plot(
        color="blue", linestyle='dashed', label='Confidence interval')
    plt.title("Global sea-level anomalies")
    plt.ylabel("Sea level anomaly (m)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outpath, 'ex1_sealevel-trend.png'))
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
    plot_sealevel(fn_sealevel, outpath)

    trend_analysis(fn_sealevel, fn_temp, fn_soi, outpath)

    # insert here the code for your own analysis ...







