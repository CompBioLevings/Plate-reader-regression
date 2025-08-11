#/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: MFI-pos-hit-regression.py
Author: Daniel Levings
Date: 2022-10-02

The purpose of this script is to read in a folder of mean fluorescence intensity
(MFI) data from a plate reader assay which is used to check for fluorescence
induced by a library of test compounds vs a set of positive and negative controls. 
I use a GLM to predict reactivity (and Lasso regression to pick the most highly 
predictive, but minimal 'factored' model, to prevent overfitting) as well as standard 
non-parametric tests comparing the negative control and test compounds.  A report
with the GLM characterists, the number of positive hits (excluding control),
and plots showing the distributions/reactivity are output for the scientist/
user.
'''

# Load necessary packages into environment
from contextlib import contextmanager
import os
import argparse
import socket
import errno
import sys
import datetime
import re
import itertools
import math
import pandas as pd
import numpy as np
import xlsxwriter
import pytz
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.stats.multitest
import scipy
import matplotlib.pyplot as plt
import plotnine as p9

# Function for generating datetime string
def datetime_str(timezone='America/Chicago'):
    """
    Return current date and time in the specified time zone as a string
    in the format: YYYYMMDD-HHMMSS.
    """
    tz = pytz.timezone(timezone)
    t = datetime.datetime.now(tz)
    y = str(t.year).zfill(4)
    m = str(t.month).zfill(2)
    d = str(t.day).zfill(2)
    h = str(t.hour).zfill(2)
    M = str(t.minute).zfill(2)
    s = str(t.second).zfill(2)
    return f"{y}{m}{d}-{h}{M}{s}"

# Commandline parsing function so this can be run via commandline
def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description = 
        """ Does quality control and identifies potential reactive compounds from a folder of plate reader MFI data.
        
        Parse set of mean flourescence intensity (MFI) values from plate reader, which includes standards 
        (positive and negative), to calculate linear model and identify positive 'hits' causing fluorescence 
        above baseline/noise. Also generates a report with the GLM characteristics, number of positive compounds 
        (excluding control), plots showing the distributions/reactivity and potential sources of noise. Normally 
        set to use logistic regression, but can be set to use linear regression (Gaussian) if desired.
        
        Example usage:
        $ python3 MFI-hit-regression.py --dir ./test-data --name test_experiment --linear
        """
    )
    
    parser.add_argument("--dir", type=str, default='./', help="Target folder to retrieve plate reader TSV files and output results.")
    
    parser.add_argument("-n", "--name", type=str, nargs='?', default=None, \
        help="Optional - give a name to the data, if not provided, will generate generic " \
        + "output name based on folder name and current date/time.")
    
    parser.add_argument("--linear", action='store_true', default=False, \
        help="Set this flag if you want to use linear regression (Gaussian) instead of " \
            + "logistic regression. Otherwise, defaults to logistic regression.")
    
    args=parser.parse_args()
    
    # Get directory
    data_dir = str(args.dir)
    
    # Check if data_dir is a valid path, and remove ending backslash if present
    if (os.path.exists(data_dir)):
        data_dir = re.sub(pattern = "/$", repl = "", string = data_dir)
    else:
        raise Exception("ERROR: You did not provide a valid directory to load in experiment data.\n")
    
    # Generate a static datetime string for use in output file names
    static_datetime_str = datetime_str()
    
    # If user didn't provide data name, generate based on folder provided:
    if (args.name is None or args.name == "None"):
        # Generate data name with just timestamp and generic results name
        data_name = 'results_' + static_datetime_str
    # otherwise, use the name provided by user
    else:
        # Make sure the name is a string
        if not isinstance(args.name, str):
            raise TypeError("ERROR: The name provided must be a string.")
        # Make sure it only uses standard alphanumeric characters, underscores and hyphens
        if not re.match(r'^[\w\-\ ]+$', args.name):
            raise ValueError("ERROR: The name provided can only contain alphanumeric characters, spaces, underscores, and hyphens.")
        # Otherwise, use the name provided by user
        data_name = str(args.name).strip()
    
    return data_dir, data_name, args.linear, static_datetime_str


'''
Define some helper functions for later in the script
'''

# Function for likelihood ratio test to compare two models
def likelihood_ratio(llmin, llmax, df):
    LR=2*(llmax-llmin)
    p = scipy.stats.chi2.sf(LR, df)
    return p

# Helper function to get significant digits
def signif(x: float, digits: int):
    """
    Rounds a number to specified number of significant digits.
    
    Args:
    x (float): the number to be rounded
    digits (int): the number of significant digits to retain
    
    Returns:
    (float): the number rounded to the specified number of significant digits
    """

    x = float(x)
    digits = int(digits)
    return round(x, -int(math.floor(math.log10(abs(x)))) + (digits - 1))

# Function to get interaction terms for complicated models
def get_design_with_pair_interaction(data, group_pair):
    """ Get the design matrix with the pairwise interactions
    
    Attribution:
    https://www.anycodings.com/1questions/920595/interactions-between-dummies-variables-in-python
    
    Parameters
    ----------
    data (pandas.DataFrame):
        Pandas data frame with the two variables to build the design matrix of their two main effects and their interaction
    group_pair (iterator):
        List with the name of the two variables (name of the columns) to build the design matrix of their two main effects and their interaction
    
    Returns
    -------
    x_new (pandas.DataFrame):
        Pandas data frame with the design matrix of their two main effects and their interaction
    
    """
    x = pd.get_dummies(data[group_pair])
    interactions_lst = list(
        itertools.combinations(
            x.columns.tolist(),
            2,
        ),
    ) 
    x_new = x.copy()
    for level_1, level_2 in interactions_lst:
        if level_1.split('_')[0] == level_2.split('_')[0]:
            continue
        x_new = pd.concat(
            [
                x_new,
                x[level_1] * x[level_2]
            ],
            axis=1,
        )
        x_new = x_new.rename(
            columns = {
                0: (level_1 + '_' + level_2)
            }
        )
    return x_new

# Function to suppress warnings and messages to console
@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def main():
    try:
        '''
        Retrieve user specified data
        '''
        data_dir, data_name, use_linear, static_datetime = parse_args()
        

        '''
        First I need to load in the data from the experiment.  The folder with the data
        is provided by the user, and I go to that folder, search for all tab-separated
        files (.tsv) within all subfolders that are named "Experiment_x_Plate_y", 
        where x and y are numbers, read in all the data from these, and convert to
        a Pandas dataframe for doing calculations on.
        '''
        
        # Check if valid path, and remove ending backslash if present
        if (os.path.exists(data_dir)):
            pass
        elif(os.path.exists(str(os.getcwd()) + "/" + data_dir)):
            data_dir = re.sub(pattern = "/$", repl = "", string = str(os.getcwd()) + "/" + data_dir)
        else:
            raise Exception("ERROR: You did not provide a valid directory to load in experiment data.\n")
                
        # Make list variables to store data attributes in
        data_paths = []
        data_names = []
        experiment_num = []
        plate_num = []
        
        # Make regex expression to identify all data files, assign appropriate names to samples, 
        # and store data from name in variables to add back to dataframes later.
        regex_data = re.compile(r"(Experiment_(\d+)_Plate_(\d+)).tsv", re.IGNORECASE)
        for root, dirs, files in os.walk(data_dir):
            files = np.sort(np.array(files)).tolist()
            for file in files:
                if regex_data.match(file):
                    data_paths.append(str(data_dir+"/"+file))
                    file_name = regex_data.search(file)
                    data_names.append(file_name.group(1))
                    experiment_num.append(file_name.group(2))
                    plate_num.append(file_name.group(3))
        del(root, dirs, files)
        
        # Initialize list object for reading in data
        data_object = []
        # Read in data
        for i in range(0, len(data_paths)):
            try:
                # make regex to identify header line
                regex_header = re.compile(r"name.*Runtime", re.IGNORECASE)
                with open(str(data_paths[i])) as file:
                    for line in file.readlines():
                        # Check if header and omit if so, if not, go to else: and append line
                        if regex_header.match(line):
                            continue
                        else:
                            # Add additional info to the end with tab separation and remove trailing newline
                            line_fin = str(line.strip()) + "\t" + str(experiment_num[i]) + "\t" + str(plate_num[i])
                            data_object.append(line_fin)
            except:
                raise Exception(f"ERROR: An error occurred while reading the file " \
                    + f"{ str(data_paths[i]) }.\n")
        del(file, i, line, line_fin)
        
        # Now convert to Pandas dataframe
        plate_data = pd.DataFrame([x.split('\t') for x in data_object], columns=['Cpd_name', 
            'Plate_ID', 'Well', 'MFI', 'Runtime', 'Experiment', 'Plate_ID_confirm'])
        
        # Confirm that the plate ID in dataframes matches the plate ID in filenames
        # If it all looks good- remove the redundant Plate ID column
        if any(plate_data['Plate_ID'] != plate_data['Plate_ID_confirm']):
            sys.exit("ERROR: Plate ID in filename does not match plate ID in data.\n")
        else:
            plate_data = pd.DataFrame(plate_data[['Cpd_name', 'Experiment', 'Runtime', 
                'Plate_ID', 'Well', 'MFI']], copy = True)
        
        '''
        Now that the data has been loaded, I add some additional information, and convert
        some columns into categorical data for using in generalized linear modeling.
        '''
        
        # Split the well information to get row and column separately -- to check
        # for differences in signal based on where the well is on the plate
        plate_data['Well_row'] = [re.sub(pattern = "[0-9]+$", repl = "", string = x) for x in plate_data['Well']]
        plate_data['Well_col'] = [re.sub(pattern = "^[A-H]", repl = "", string = x) for x in plate_data['Well']]
        
        # Now split time and data info
        plate_data['Date'] = pd.to_datetime(plate_data['Runtime']).dt.date
        plate_data['Time'] = pd.to_datetime(plate_data['Runtime']).dt.time
        
        # Now, make a backup copy of the plate_data, so I can keep the original and use
        # plate_data to manipulate for modeling, and keep the original 
        plate_data_orig = pd.DataFrame(plate_data, copy = True)
        # plate_data = pd.DataFrame(plate_data_orig, copy = True)
        
        # Convert well column to two-digit, and then replace original 'Well' column
        plate_data['Well_col'] = ["%02d" % int(x) for x in plate_data['Well_col']]
        plate_data['Well'] = plate_data['Well_row'] + plate_data['Well_col']
        
        # Now add row and col in front of location, and plate and experiment
        # in front of their strings/numbers
        plate_data['Well_col'] = "Col_" + plate_data['Well_col']
        plate_data['Well_row'] = "Row_" + plate_data['Well_row']
        
        # Now make Well, Well_row, Well_col, Plate_ID, Experiment and Date all
        # categorical variables for including in modeling
        plate_data['Well'] = pd.Categorical(plate_data['Well'], ordered = True)
        plate_data['Well_row'] = pd.Categorical(plate_data['Well_row'], ordered = True)
        plate_data['Well_col'] = pd.Categorical(plate_data['Well_col'], ordered = True)
        plate_data['Plate_ID'] = pd.Categorical(plate_data['Plate_ID'], ordered = True)
        plate_data['Experiment'] = pd.Categorical(plate_data['Experiment'], ordered = True)
        plate_data['Date'] = pd.Categorical(plate_data['Date'], ordered = True).codes
        
        # Do the same with Date, but convert to values (integers)
        plate_data['Date'] = ["Date_" + str(x) for x in pd.Categorical(plate_data['Date'], ordered = True).codes.tolist()]
        
        # Make MFI a numerica (not string)
        plate_data['MFI'] = pd.to_numeric((plate_data['MFI']))
        
        # Convert time to integer (in seconds)
        time_int = []
        for hhmmss in plate_data['Time']:
            [hours, minutes, seconds] = [int(x) for x in str(hhmmss).split(':')]
            x = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
            time_int.append(x.seconds)
        plate_data['Time_int'] = time_int
        del(hours, minutes, seconds, x, time_int, hhmmss)
        
        # Then also, get the start time of each experiment and plate, and subtract that 
        # start time from the Time_int to get how long each well was 'sitting' waiting to
        # be read by the plate reader (assuming each plate was prepared just before reading)
        min_exp_times = plate_data[['Time_int', 'Plate_ID', 'Experiment']].groupby(\
            ['Experiment', 'Plate_ID'], observed=False).agg('min')
        min_exp_times.reset_index(inplace = True)
        min_exp_times = min_exp_times.rename(columns = {'Time_int' : "Time_min"})
        
        # Now combine this back with the plate_data, and convert Time_int to subtract
        # the min time of each experiment
        plate_data = plate_data.merge(how = 'left', right = min_exp_times)
        plate_data['Time_int'] = plate_data['Time_int'] - plate_data['Time_min']
        plate_data = pd.DataFrame(plate_data.drop(columns = 'Time_min'), copy = True)
        
        '''
        Now, I extract the control data (postive and negative) and make a linear model with 
        the control data to make a 'standard curve', and also to check if any of the other 
        factors in the data are impacting the model. (if there's any *artefactual* signal
        to "subtract" from the data)
        '''
        
        # Extract only the positive and negative control data to use to 'build' GLM
        control_plate_df = pd.DataFrame(plate_data.loc[plate_data['Cpd_name'].isin(['NOCPD', 
            'CPD_00000']),], copy = True)
        
        # Turn the pos and negative into 0 and 1 respectively, as categorical variables
        control_plate_df['Cpd_name'] = pd.Categorical(control_plate_df['Cpd_name'], 
            categories = ['NOCPD', 'CPD_00000'], ordered = True)
        
        # Check this column
        # control_plate_df['Cpd_name']
        
        # Not enough data points to incorporate well row or column into confounding variables
        
        # Test if experiments run on different dates, exclusively, and if so, don't
        # include the Date dummies for GLM
        if (control_plate_df.drop_duplicates(subset = ['Experiment', 'Date']).shape[0] == 
            len(np.unique(control_plate_df['Experiment']))):
            # Now convert most of the categorical columns into binary data for modeling -
            # use 'get_dummies' to convert each into binary, splitting into columns based on 
            # the factor levels present, and then concatenate all these into a single large
            # dataframe for the GLM - don't include GLM
            control_plate_GLM = pd.concat([get_design_with_pair_interaction(data = control_plate_df, 
                group_pair = ['Experiment', 'Plate_ID']), pd.get_dummies(control_plate_df['Cpd_name']), 
                control_plate_df[['Time_int', 'MFI']]], axis = 1)
        else:
            # Make dummies including Dates for GLM if Experiment != Date
            control_plate_GLM = pd.concat([get_design_with_pair_interaction(data = control_plate_df, 
                group_pair = ['Experiment', 'Plate_ID']), pd.get_dummies(control_plate_df['Date']), 
                pd.get_dummies(control_plate_df['Cpd_name']), control_plate_df[['Time_int', 'MFI']]], axis = 1)
        
        # Remove the NOCPD column since it's redundant with the CPD_00000
        control_plate_GLM = pd.DataFrame(control_plate_GLM.drop(columns = 'NOCPD'), copy = True)
        control_plate_GLM.reset_index(drop = True, inplace = True)
        
        # Now scale/normalize the numeric columns so they work better with GLM
        # list which columns have numerical data
        # control_plate_GLM.info()
        numerical_cols = ['MFI', 'Time_int']
        
        # Do some data normalization/scaling on the numeric variables (not dummies)
        scaler = StandardScaler().fit(control_plate_GLM[numerical_cols])
        control_plate_GLM[numerical_cols] = scaler.transform(control_plate_GLM[numerical_cols])
        
        # write control_plate_GLM to a file for debugging
        # control_plate_GLM.to_csv("control_plate_GLM.tsv", index = False, sep = "\t")
        
        # Now make a GLM with these data - first reduced, with no 'extra' variables
        # and then a full model with all the other potential 'confounders' - note:
        # I want to see if MFI can *predict* control compound type
        if not use_linear:
            min_model = sm.GLM.from_formula(formula = "MFI ~ CPD_00000", 
                data = control_plate_GLM.astype('float'), family = sm.families.Binomial())
        else:
            min_model = sm.GLM.from_formula(formula = "MFI ~ CPD_00000", 
                data = control_plate_GLM.astype('float'), family = sm.families.Gaussian())
        min_res = min_model.fit()
        
        # Extract only the log-likelihood value and fitted values
        # dir(min_res)
        min_llf = getattr(min_res, 'llf')
        min_fittedvals = pd.to_numeric(getattr(min_res, 'fittedvalues'))
        
        # Set theme for plotting
        p9.themes.theme_set(p9.theme_bw(base_family = "Sans Serif") + 
            p9.theme(plot_title = p9.element_text(ha="center", va="baseline", 
                    size = 14, margin = {'t' : 5, 'b' : 0, 'l' : 5, 'r' : 5}), 
                axis_text = p9.element_text(color="black", size = 11), 
                axis_title = p9.element_text(color="black", size = 12), 
                axis_ticks = p9.element_line(color="black"), 
                legend_title = p9.element_text(face = "italic", size = 12),
                legend_text = p9.element_text(size = 11),
                panel_border = p9.element_rect(size=0.75, color="black"), 
                legend_position = "right",
                panel_grid_major = p9.element_line(color = "silver", linetype="solid", size=0.2),
                panel_grid_minor = p9.element_line(color = "silver", linetype="dashed", size=0.1)
            ))
        
        # Make plot of the linear model/fit with only 'reduced' variables
        control_GLM_plot = (
            p9.ggplot(data = pd.concat([control_plate_GLM.assign(control_cat = 
                pd.Categorical(control_plate_GLM['CPD_00000'], ordered = True)), 
                pd.DataFrame(min_fittedvals).rename(columns = {0 : "fitted"})], axis = 1), 
                mapping = p9.aes(y="MFI", x = "fitted", color="control_cat")) + 
            p9.geom_point(size = 3, alpha = 0.7) + 
            p9.geom_vline(xintercept = 0.5, linetype = "dashed") +
            p9.scales.scale_color_manual(values = {0 : "royalblue" , 1 : "red"}) +
            p9.labs(x = "Fitted values from GLM prediction", y = "Scaled MFI", 
                    title = "Minimal GLM of control data",
                    color = "Neg or pos\ncontrol (0/1)\n") +
            p9.theme(figure_size = (7, 5.5), dpi = 150)
            )
        # control_GLM_plot.show()
        
        '''
        Now make GLM with full model, including potential confounders and interactors.  This time,
        I have potential for overfitting and including too many variables.  So I will
        split into testing and training sets and use LASSO regression with CV 
        from sklearn to optimize the model selection
        '''
        
        # First split output and input variables - y = output, X = input
        y = control_plate_GLM['CPD_00000']
        
        # Make input data into new dataframe for input
        X = pd.DataFrame(control_plate_GLM.drop(columns = ['CPD_00000']), copy = True)
        
        # Now do LASSO regression with 5-fold cross validations
        full_model = LassoCV(cv=5, max_iter = 10000, random_state=0)
        full_res = full_model.fit(X = X, y = y)
        
        # Now choose the best model using alpha from LassoCV
        full_best = Lasso(alpha = full_res.alpha_)
        full_best.fit(X, y)
        
        # dir(full_best.fit(X, y))
        # getattr(full_best, 'llf')
        
        # Check how it 'scores'
        # print(full_best.intercept_, full_best.coef_, full_best.score(X, y))
        # print(list(zip(full_best.coef_, X)))
        
        # Now extract only the variables that have non-zero coefficient (keep in model)
        keep_model_cols = []
        for i in range(0, len(full_best.coef_.tolist())):
            if full_best.coef_.tolist()[i] != 0:
                keep_model_cols.append(X.columns[i])
        del(i)
        
        # Now use these columns only to generate regression with most important vars
        if not use_linear:
            full_model = sm.GLM.from_formula(formula = str("CPD_00000 ~ " + 
                ' + '.join(keep_model_cols)), data = control_plate_GLM.astype('float'), family = sm.families.Binomial())
        else:
            full_model = sm.GLM.from_formula(formula = str("CPD_00000 ~ " + 
                ' + '.join(keep_model_cols)), data = control_plate_GLM.astype('float'), family = sm.families.Gaussian())
        full_res = full_model.fit()
        
        # Check which variables had significant impact on model
        # dir(full_res)
        # get_pvals = pd.DataFrame(getattr(full_res, 'pvalues').sort_values()
        #     ).reset_index().rename(columns = {'index' : "colname", 0 : "pval"})
        # keep_model_cols = get_pvals.loc[(get_pvals['pval'] < 0.05) & (get_pvals["colname"] != "Intercept"),
        #     "colname"].tolist()
        
        # Extract only the log-likelihood value and fitted values
        full_llf = getattr(full_res, 'llf')
        full_fittedvals = pd.to_numeric(getattr(full_res, 'fittedvalues'))
        
        # Now compare the full and reduced models by p-value
        degrees_of_freedom = len(keep_model_cols)-1 
        full_vs_min_LRT = likelihood_ratio(llmin = min_llf, llmax = full_llf, df = degrees_of_freedom)
        
        # Make plot of the linear model/fit with all variables
        control_GLM_plot_full = (
            p9.ggplot(data = pd.concat([control_plate_GLM.assign(control_cat = 
                pd.Categorical(control_plate_GLM['CPD_00000'], ordered = True)), 
                pd.DataFrame(full_fittedvals).rename(columns = {0 : "fitted"})], axis = 1), 
                mapping = p9.aes(y="MFI", x = "fitted", color="control_cat")) + 
            p9.geom_point(size = 3, alpha = 0.7) + 
            p9.geom_vline(xintercept = 0.5, linetype = "dashed") +
            p9.scales.scale_color_manual(values = {0 : "royalblue" , 1 : "red"}) +
            p9.labs(x = "Fitted values from GLM prediction", y = "Scaled MFI", 
                title = "Full GLM of control data - LRT p-val = " + 
                str(signif(full_vs_min_LRT, digits = 3)), color = "Neg or pos\ncontrol (0/1)\n") +
            p9.theme(figure_size = (7, 5.5), dpi = 150)
            )
        # control_GLM_plot_full.show()
        
        '''
        Now I have a working model that clearly separates the positive and negative
        controls.  I will use the coefficients from the LASSO regression model to
        predict the status of the 'test' data (the compounds where it is unknown if
        they are a positive 'hit'/reactive compound or not).
        
        '''
        
        # First I have to generate a dataframe of the relevant columns and dummies from the 
        # full dataset (including positive and negative controls)
        
        # Again, only include Date if it doesn't perfectly overlap with Experiment
        if (control_plate_df.drop_duplicates(subset = ['Experiment', 'Date']).shape[0] == 
            len(np.unique(control_plate_df['Experiment']))):
            # Now convert most of the categorical columns into binary data for modeling -
            # use 'get_dummies' to convert each into binary, splitting into columns based on 
            # the factor levels present, and then concatenate all these into a single large
            # dataframe for the GLM - don't include GLM
            test_data = pd.concat([get_design_with_pair_interaction(data = plate_data, 
                group_pair = ['Experiment', 'Plate_ID']), 
                plate_data[['Time_int', 'MFI', 'Cpd_name']]], axis = 1)
        else:
            # Make dummies including Dates for GLM if Experiment != Date
            test_data = pd.concat([get_design_with_pair_interaction(data = plate_data, 
                group_pair = ['Experiment', 'Plate_ID']), pd.get_dummies(plate_data['Date']), 
                plate_data[['Time_int', 'MFI', 'Cpd_name']]], axis = 1)
        
        # Now scale the MFI and Time data for fitting - backup originals first
        test_data['MFI_unscaled'] = test_data['MFI']
        test_data['Time_int_unscaled'] = test_data['Time_int']
        
        # Now normalize - using previous scale
        test_data[['MFI', 'Time_int']] = scaler.transform(test_data[['MFI', 'Time_int']])
        
        # Now use the model from the Lasso regression of above to predict the 
        # status of the novel compounds
        test_res = pd.concat([test_data, pd.DataFrame(full_best.predict(test_data.drop(columns = 
            ["Cpd_name", "MFI_unscaled", "Time_int_unscaled"]))).rename(columns = {0 : 'fitted'})], axis = 1)
        
        # Add column for whether it's positive or negative control, or test
        control_or_test = []
        for i in test_res['Cpd_name']:
            if (i == "NOCPD"):
                control_or_test.append("Negative")
            elif (i == "CPD_00000"):
                control_or_test.append("Positive")
            else:
                control_or_test.append("Test")
        test_res['Control_v_test'] = control_or_test
        del(i, control_or_test)
        
        # # Make linear plot like before, but including test data
        # plot_test_data = (
        #     p9.ggplot(data = test_res, 
        #         mapping = p9.aes(y="MFI_unscaled", x = "fitted", color="Control_v_test")) + 
        #     p9.geom_point(size = 3, alpha = 0.7) + 
        #     p9.geom_vline(xintercept = 0.5, linetype = "dashed") +
        #     p9.scales.scale_color_manual(values = {'Negative' : "royalblue", 
        #         'Positive' : "red", 'Test' : "gray"}) +
        #     p9.labs(x = "Reactivity from GLM fit (0 = low, 1 = high)", y = "MFI", 
        #             title = "Full GLM of control and test data",
        #             color = "Control or Test") +
        #     p9.theme(figure_size = (7, 5.5), dpi = 350)
        #     )
        
        # Make plot showing 'split' of data and full variation - combining sina plot
        # which is 2d density plot, and box plots, to show spread of data
        plot_test_data = (
            p9.ggplot(data = test_res, 
                mapping = p9.aes(x="Control_v_test", y = "fitted", color="Control_v_test")) + 
            p9.geom_sina(maxwidth = 0.9, size = 2.5,
                method = "density", scale = "width", alpha = 0.6) + 
            p9.geom_boxplot(color = "black", size = 0.7, alpha = 0, width = 0.4, outlier_alpha = 0) +
            p9.geom_hline(yintercept = 0.5, linetype = "dashed") +
            p9.scales.scale_color_manual(values = {'Negative' : "royalblue", 
                'Positive' : "red", 'Test' : "gray"}) +
            p9.labs(y = "Reactivity from GLM fit (0 = low, 1 = high)", x = "Category", 
                    title = "GLM-derived reactivity of control and test data (with reps)",
                    color = "Control or Test") +
            p9.theme(figure_size = (7, 5.5), dpi = 350)
            )
        
        # Now consolidate get mean values across experiments and plates for each compound,
        # and plot these average values
        summ_df = test_res[['Cpd_name', 'MFI', 'MFI_unscaled', 'fitted']].groupby('Cpd_name').agg('mean')
        summ_df.reset_index(inplace = True)
        
        # Now classify as an compound that is reacting in the test (a postiive 'hit') or not,
        # by partitioning to 0 or 1 from fitted value
        summ_df['Reactive'] = round(summ_df['fitted'], 0)
        
        # It's possible some had over 1 even for 'Reactive', so I will correct that
        reactive = []
        for i in summ_df['Reactive']:
            if (i > 1):
                reactive.append(1)
            elif (i < 0):
                reactive.append(0)
            else:
                reactive.append(i)
        summ_df['Reactive'] = reactive
        del(i, reactive)
        
        # Assign to control or test again
        control_or_test = []
        for i in summ_df['Cpd_name']:
            if (i == "NOCPD"):
                control_or_test.append("Negative")
            elif (i == "CPD_00000"):
                control_or_test.append("Positive")
            else:
                control_or_test.append("Test")
        summ_df['Control_v_test'] = control_or_test
        del(i, control_or_test)
        
        # Now plot this summary info
        # Reverse sort by if it is control data, for plotting
        summ_df.sort_values(['Control_v_test', 'fitted'], ascending = False, inplace = True)
        
        # Make plot of the linear model/fit with all variables
        plot_summ_data = (
            p9.ggplot(data = summ_df, 
                mapping = p9.aes(x="Control_v_test", y = "fitted", color="Control_v_test")) + 
            p9.geom_sina(maxwidth = 1.1, size = 2.5,
                method = "density", scale = "area", alpha = 0.6) + 
            p9.geom_boxplot(color = "black", size = 0.7, alpha = 0, width = 0.4, outlier_alpha = 0) +
            p9.scales.scale_color_manual(values = {'Negative' : "royalblue", 
                'Positive' : "red", 'Test' : "gray"}) +
            p9.geom_hline(yintercept = 0.5, linetype = "dashed") +
            p9.labs(y = "Average Reactivity from GLM fit (0 = low, 1 = high)", x = "Category", 
                    title = "GLM-derived reactivity of control and test data - summarized",
                    color = "Control or Test") +
            p9.theme(figure_size = (7, 5.5), dpi = 350)
            )
        
        # Now sort summary dataframe by whether compound is predicted to be reactive or not,
        # and then by the fitted values 
        summ_df.sort_values(['Reactive', 'fitted'], ascending = False, inplace = True)
        
        # Now calculate the log2(fold-change) in signal between the data points and the
        # negative control for: 1) Average of fitted values, 2) Average of raw MFI values
        # First for fitted values - need to add a constant to prevent getting negative or
        # 0 log values
        add_min_fit = math.floor(min(summ_df['fitted'].tolist()))*-1
        neg_control_fit = float(summ_df.loc[summ_df['Control_v_test'] == "Negative", "fitted"])
        summ_df['log2FC_predicted'] = [math.log2(x + add_min_fit) - math.log2(neg_control_fit + add_min_fit) for x in summ_df['fitted']]
        
        # Now for average MFI
        add_min_MFI = math.floor(min(summ_df['MFI_unscaled'].tolist()))*-1
        
        # Subtract 1 if add_min_MFI == one of the MFI values (to prevent log2(0) error)
        if any([(float(add_min_MFI*-1) == x) for x in summ_df['MFI_unscaled']]):
            add_min_MFI = add_min_MFI+1
        
        # Now compute log2(fold-change)
        neg_control_MFI = float(summ_df.loc[summ_df['Control_v_test'] == "Negative", "MFI_unscaled"])
        summ_df['log2FC_MFI'] = [math.log2(x + add_min_MFI) - math.log2(neg_control_MFI + add_min_MFI) for x in summ_df['MFI_unscaled']]
        
        '''
        Now do more tests of which compounds show significant signal by doing more
        "basic" non-parametric statistical tests (using Mann-Whitney rank test) because
        these data show non-normal distribution (many cluster around negative control and
        a few compounds are found by positive control- so long tail).  I will specifically use
        a one-sided test to see if the compound show significantly *higher* MFI than
        the negative control
        '''
        
        # First, isolate only values for negative control
        neg_MFIs = np.array(plate_data.loc[plate_data['Cpd_name'] == "NOCPD", "MFI"])
        
        # Now make dataframe for iterating through to get p-values for every compound
        # but the negative control
        stat_df = pd.DataFrame({"Cpd_name" : summ_df.loc[summ_df['Cpd_name'] != "NOCPD", "Cpd_name"].unique()})
        stats_list = []
        for i in stat_df['Cpd_name'].tolist():
            # i = "CPD_00000"
            test = np.array(plate_data.loc[plate_data['Cpd_name'] == i, "MFI"])
            stat_res = scipy.stats.mannwhitneyu(y = neg_MFIs, x = test, alternative = "greater")
            stats_list.append(getattr(stat_res, 'pvalue')) 
            dir(stat_res)
        stat_df['pval'] = stats_list
        
        # Do multiple-hypothesis testing correction
        FDR_bool, FDR_list = statsmodels.stats.multitest.fdrcorrection(pvals = stats_list)
        
        # Add back to stat_df
        stat_df['FDR'] = FDR_list
        stat_df['FDR-&-GLM_pass'] = FDR_bool
        
        # Cleanup
        del(i, neg_MFIs, test, stat_res, FDR_bool, FDR_list)
        
        # Finally, combine the Mann-Whitney results back into the main result dataframe
        summ_df = summ_df.merge(how = 'left', right = stat_df)
        
        # Put together FDR and GLM booleans to make final value for 'FDR-&-GLM_pass'
        FDR_GLM_pass = []
        for i in range(0, len(summ_df['FDR-&-GLM_pass'])):
            # if else to skip the negative control
            if np.isnan(summ_df.loc[i,'FDR-&-GLM_pass']):
                FDR_GLM_pass.append(False)
            else:
                FDR_GLM_pass.append(bool(summ_df.loc[i,'FDR-&-GLM_pass'] & (summ_df.loc[i,'Reactive'] == 1)))
        summ_df['FDR-&-GLM_pass'] = FDR_GLM_pass
        del(i, FDR_GLM_pass)
        
        # Resort by passing both this and LRT
        summ_df.sort_values(['FDR-&-GLM_pass', 'fitted'], ascending = False, inplace = True)
        
        '''
        Now that I have some results, I also want to do some QC to see if there are any
        artefactual signals that the scientists should be aware of in the assay.  So, I will
        plot the MFI values relative to individual Plate and Experiment and also relative to
        any of the columns retained in the final/full Lasso GLM model.
        '''
        
        # First just do MFI by plate and experiment
        # Add column for whether it's positive or negative control, or test
        control_or_test = []
        for i in plate_data['Cpd_name']:
            if (i == "NOCPD"):
                control_or_test.append("Negative")
            elif (i == "CPD_00000"):
                control_or_test.append("Positive")
            else:
                control_or_test.append("Test")
        plate_data['Control_v_test'] = pd.Categorical(control_or_test, ordered = True)
        del(i, control_or_test)
        
        # Now do some QC steps
        # First- plot experiment and plate against MFI value to see if there is 
        # significant variation - make plot
        QC_plate_data = plate_data.assign(Exp_Plate = pd.Categorical([("Exp_" + 
            str(plate_data.loc[i,'Experiment']) + "\nPlate_" + str(plate_data.loc[i,'Plate_ID'])) 
            for i in range(0,len(plate_data['Experiment']))])).sort_values(['Control_v_test'], 
            ascending = False)
        
        # Plot
        plot_exp_plate_data = (
            p9.ggplot(data = QC_plate_data, mapping = p9.aes(x="Exp_Plate", y = "MFI", 
                group = "Exp_Plate", color="Control_v_test")) + 
            p9.geom_sina(maxwidth = 0.8, size = 2.5, 
                method = "density", scale = "width", alpha = 0.6) + 
            p9.geom_boxplot(color = "black", size = 0.7, alpha = 0, width = 0.3, outlier_alpha = 0) +
            p9.scales.scale_color_manual(values = {'Negative' : "royalblue", 
                'Positive' : "red", 'Test' : "gray"}) +
            p9.labs(y = "MFI", x = "Experiment and plate", 
                title = "MFI by experiment and plate",
                color = "Control or Test") +
            p9.theme(figure_size = (2 + len(pd.unique(QC_plate_data['Exp_Plate'].values)), 5.5), 
                dpi = 350) 
            # make plot wider based on the number of plate-experiment combinations
            )
        
        # Now, start putting all the plots into a plotting object (list)
        # Initialize list
        plots_object = []
        
        # Add previous plots to it
        # First- the overall trends of reactivity from the summarized (mean) MFI
        plots_object.append(plot_summ_data)
        
        # Now with independent reps shown
        plots_object.append(plot_test_data)
        
        # Now, add QC plot- looking at Experiments and Plates
        plots_object.append(plot_exp_plate_data)
        
        # Add GLM minimal and full plots to plot object
        plots_object.append(control_GLM_plot)
        plots_object.append(control_GLM_plot_full)
        
        # Now plot all experimental factors included in final Lasso model
        # First remove the dependent variable (MFI)
        plot_factors = np.array(keep_model_cols)[np.array([x != "MFI" for x in keep_model_cols])].tolist()
        
        # extract what type of factor they are (categorical/dummy, or numeric)
        float_factors = np.array(plot_factors)[np.array([pd.api.types.is_float_dtype(x) 
            for x in test_res[plot_factors].dtypes])].tolist()
        cat_factors = np.array(plot_factors)[np.invert(np.array([pd.api.types.is_float_dtype(x) 
            for x in test_res[plot_factors].dtypes]))].tolist()
        
        # Make new dataframe for plotting
        test_res_tmp = pd.DataFrame(test_res.sort_values('Control_v_test', ascending = False), copy = True)
        
        # Now plot the categorical variables
        if (len(cat_factors) > 0): 
            for i in range(0, len(cat_factors)):
                # i = 0
                # Now also make plots looking at specific factors in model
                test_res_tmp[cat_factors[i]] = pd.Categorical(test_res_tmp[cat_factors[i]], ordered = True)
                plot_model_factor_data = (
                    p9.ggplot(data = test_res_tmp, 
                        mapping = p9.aes(x = cat_factors[i], y = "MFI_unscaled", 
                        group = cat_factors[i], color="Control_v_test")) + 
                    p9.scale_x_discrete() +
                    p9.geom_sina(maxwidth = 0.8, size = 2.5, 
                        method = "density", scale = "width", alpha = 0.6) + 
                    p9.geom_boxplot(color = "black", size = 0.7, alpha = 0, width = 0.3, outlier_alpha = 0) +
                    p9.scales.scale_color_manual(values = {'Negative' : "royalblue", 
                        'Positive' : "red", 'Test' : "gray"}) +
                    p9.labs(y = "MFI", x = cat_factors[i] + " (0 = no, 1 = yes)", 
                        title = "MFI by " + cat_factors[i],
                        color = "Control or Test") +
                    p9.theme(figure_size = (4 + len(pd.unique(test_res[cat_factors[i]].values).tolist()), 5.5), 
                        dpi = 350) 
                    # make plot wider based on the number of plate-experiment combinations
                    )
                
                # Add to plots object
                plots_object.append(plot_model_factor_data)
        
        # Now plot any numeric factors in model
        if (len(float_factors) > 0): 
            for i in range(0, len(float_factors)):
                # i = 0
                # Now also make plots looking at specific factors in model
                plot_model_factor_data = (
                    p9.ggplot(data = test_res_tmp, 
                        mapping = p9.aes(x = float_factors[i] + "_unscaled", y = "MFI_unscaled", 
                        color="Control_v_test")) + 
                    p9.geom_point(alpha = 0.6) + 
                    p9.geom_smooth(alpha = 0, method = "glm") +
                    p9.scales.scale_color_manual(values = {'Negative' : "royalblue", 
                        'Positive' : "red", 'Test' : "gray"}) +
                    p9.labs(y = "MFI", x = float_factors[i], 
                        title = "MFI by " + float_factors[i],
                        color = "Control or Test") +
                    p9.theme(figure_size = (7, 5.5), 
                        dpi = 350) 
                    # make plot wider based on the number of plate-experiment combinations
                    )
                
                # Add to plots object
                plots_object.append(plot_model_factor_data)
        del(i)
        
        '''
        Now, I will take all the different useful information/graphics generated and
        output them to various files for the scientist to view.  These include: 
        1) all relevant plots, including reactivity, GLM comparisons, and QC, 
        2) the model summaries from the GLM fitting of minimal and full model, and
        3) an excel file of the finished results.
        '''
        
        # Now look for any strange 
        # Save all plots to single 'report' PDF
        with suppress_stderr():
            p9.save_as_pdf_pages(plots = plots_object,
                filename = data_dir + "/" + data_name + "_MFI-signal_and_QC_plots.pdf")
        
        # Calculate the number of positive compounds (minus positive control) and
        # list them.
        cpd_hits = summ_df.loc[summ_df['FDR-&-GLM_pass'] & (summ_df['Cpd_name'] != "CPD_00000"), 'Cpd_name'].tolist()
        
        # First, get original stdout
        original_stdout = sys.stdout
        
        # Now open text file and redirect stdout to text file
        with open(data_dir + "/" + data_name + "_GLM_and_system_report.txt", 'w') as file:
            sys.stdout = file # Change standard output to file
            # Output the linear model summaries and Python version info to text report
            print("Python version info: " + str(sys.version_info))
            print("Modules/packages used in script: os, sys, contextlib, datetime, re, itertools, math, pandas, numpy, sklearn, statsmodels, scipy, plotnine.\n\n")
            print("There were " + str(len(cpd_hits)) + " reactive compounds that passed both LRT and pval criteria. They are:\n" +
                  ', '.join(cpd_hits) + "\n\n")
            print("Likelihood ratio test between minimal model and full model, pval = " + str(signif(full_vs_min_LRT, 4)) + "\n")
            print("GLM Summary for minimal model:\n")
            print(min_res.summary())
            print("\n")
            print("GLM Summary for full model:\n")
            print(full_res.summary())
            sys.stdout = original_stdout # return to original state
        del(file)
        
        # Finally, output all the data from the assay, with summarized signals vs neg control
        # and statistics, to an Excel file
        
        # create metadata for Excel file (date, name, etc)
        filename = f"{ data_dir }/{ data_name }_compound_MFI-summary.xlsx"
        
        # write summary data to file
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        # Convert boolean to text for PASS column
        summ_df['FDR-&-GLM_pass'] = summ_df['FDR-&-GLM_pass'].map({True : "TRUE", False : "FALSE"})
        summ_df.drop(['MFI', 'Reactive'], axis = 1).rename(columns = {'MFI_unscaled' : "MFI_mean", 
            'fitted' : "Pred_reactivity"}).to_excel(excel_writer=writer, 
            sheet_name="Sheet1", index=False, float_format = "%.4f")
        workbook = writer.book
        
        # Set up custom formats
        NoFORMATsig4 = workbook.add_format({'num_format' : '0.0000', 'font_name' : 'Arial', 'font_size' : 11})
        NoFORMATsig2 = workbook.add_format({'num_format' : '0.00', 'font_name' : 'Arial', 'font_size' : 11})
        NoFORMATsci = workbook.add_format({'num_format': '0.00E+00', 'font_name' : 'Arial', 'font_size' : 11})
        NoFORMATtext = workbook.add_format({'num_format' : '@', 'font_name' : 'Arial', 'font_size' : 11})
        
        # Set up worksheet
        worksheet = writer.sheets['Sheet1']
        
        # Set column formatting
        worksheet.set_column(0, 0, 16, NoFORMATtext)
        worksheet.set_column(1, 1, 16, NoFORMATsig2)
        worksheet.set_column(2, 2, 14, NoFORMATsig4)
        worksheet.set_column(3, 3, 12, NoFORMATtext)
        worksheet.set_column(4, 5, 14, NoFORMATsig4)
        worksheet.set_column(6, 7, 12, NoFORMATsci)
        worksheet.set_column(8, 8, 10, NoFORMATtext)
        
        # close Excel writer
        with suppress_stderr():
            writer.close()
        
        del(writer)
    except socket.error as e:
        if e.errno != errno.EPIPE:
                # Not a broken pipe
                raise
        sys.exit(1)  # Python exits with error code 1 on EPIPE

# run it
if __name__ == '__main__':
    main()
