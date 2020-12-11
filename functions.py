# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# %%
#Basic function to explore the given pandas dataframe
def explore_data(df):
    """
    ***prints all basic infos about a dataframe
    
    INPUTS:
        df: any pandas dataframe
    
    """
    #get basic shape of dataframe
    shape = df.shape
    
    #check for dtypes and missing values
    info =df.info()
    
    #check numeric variables in dataframe
    description = df.describe()
    
    if 'sns' not in dir():
        import seaborn as sns
        sns.set()
        #check correlation of numeric variables in dataframe
        heatmap = sns.heatmap(df.corr(), annot=True, fmt='.2f')
    else:
        heatmap = sns.heatmap(df.corr(), annot=True, fmt='.2f')
    
    return shape, info, description, heatmap 


# %%
#plot single bar chart
def plt_bar_single_column(column_name):
    """
    ***Plots bar chart of single column of any pandas dataframe***
    
    INPUTS: 
        columns_name: input name of pandas column as string type
        
    OUPUTS:
        bar chart of value_counts of given column
    """
    
    #Provide a pandas series of the counts for each Professional status
    status_vals = df[column_name].value_counts()

    # The below should be a bar chart of the proportion of individuals in each professional category if your status_vals
    # is set up correctly.
    (status_vals/df.shape[0]).plot(kind="bar")
    plt.title(column_name)


# %%
#generic function for plotting bar charts which look decent too
#src: https://scentellegher.github.io/visualization/2018/10/10/beautiful-bar-plots-matplotlib.html
def create_bar_v_plot(x_values, y_values, x_label, chart_title, fig_width=10, fig_height=6, **kwargs):
    """
    ***Function for creating nice looking bar charts with vertical figs***
    
    INPUTS:
        x_values: column in dataframe to be plotted on x-axis
        y_values: column in dataframe to be plotted on y-axis
        x_label: Label for x-axis
        chart_title: title for whole bar chart
        figwidth: width of each figure in the bar chart, default is 10
        figheight: height of each figure in the bar chart, default is 6
        **kwargs: possebility to add further variables
    
    OUTPUTS:
        barplot: type of chart is bar chart
        fig: the figure in the chart
        ax: axis-element, which is necessary for running the autolabel-function
    """
    
    fig, ax = plt.subplots(figsize=(fig_width,fig_height))
    
    # create for each expense type an horizontal line that starts at x = 0 with the length 
    # represented by the specific expense percentage value.
    #fig_range=list(range(1,len(y_values.index)+1))
    #plt.hlines(y=fig_range, xmin=0, xmax=max(x_values), color='#007ACC', alpha=0.2, linewidth=5)
    
    #src: http://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    plt.style.use('fivethirtyeight')
    
    # create for each expense type a dot at the level of the expense percentage value
    #plt.plot(x_values, y_values, "o", markersize=5, color='#007ACC', alpha=0.6)
    
    # set labels
    ax.set_xlabel(x_label, fontsize=15, fontweight='black', color='#333F4B')
    
    # set axis
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # add an horizonal label for the y axis 
    #fig.text(-0.23, 0.96, y_label, fontsize=15, fontweight='black', color = '#333F4B')
    
    #plot the bar
    barplot=ax.bar(x_values, y_values)
    
    #set bar chart title
    ax.set_title(chart_title,fontsize=15, fontweight='black', color = '#333F4B')
    
    return barplot, fig, ax


# %%
#creating a horizontal barchart
def create_bar_h_plot(x_values, y_values, x_label, chart_title, fig_width=10, fig_height=6, **kwargs):
    """
    ***Function for creating nice looking bar charts with horizontal figs***
    
    INPUTS:
        x_values: column in dataframe to be plotted on x-axis
        y_values: column in dataframe to be plotted on y-axis
        x_label: Label for x-axis
        chart_title: title for whole bar chart
        figwidth: width of each figure in the bar chart
        figheight: height of each figure in the bar chart
        **kwargs: possebility to add further variables
    
    OUTPUTS:
        barplot: type of chart is bar chart
        fig: the figure in the chart
        ax: axis-element, which is necessary for running the autolabel-function
    """
    fig, ax = plt.subplots(figsize=(fig_width,fig_height))
    
    # create for each expense type an horizontal line that starts at x = 0 with the length 
    # represented by the specific expense percentage value.
    #fig_range=list(range(1,len(y_values.index)+1))
    #plt.hlines(y=fig_range, xmin=0, xmax=max(x_values), color='#007ACC', alpha=0.2, linewidth=5)
    
    #src: http://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    plt.style.use('fivethirtyeight')
    
    # create for each expense type a dot at the level of the expense percentage value
    #plt.plot(x_values, y_values, "o", markersize=5, color='#007ACC', alpha=0.6)
    
    # set labels
    ax.set_xlabel(x_label, fontsize=15, fontweight='black', color='#333F4B')
    
    # set axis
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    #plot the bar
    barplot=ax.barh(x_values, y_values)
    
    #set bar chart title
    ax.set_title(chart_title,fontsize=15, fontweight='black', color = '#333F4B')
    
    return barplot, fig, ax


# %%
#This example shows a how to create a grouped bar chart and how to annotate bars with labels automatically
#https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height,2)), fontsize=12,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# %%
#gratefully taken from the course: Udacity, Data Scientist
#and modified for generic use
def clean_data(df, dependant, drop_columns=[]):
    """
    ***This function cleans df using the following steps to produce X and y:
        1. Drop all the rows with no values in dependant
        2. Create X as all the columns that are not the dependant column
        3. Create y as the dependant column
        4. Drop the selected columns from X
        5. For each numeric variable in X, fill the column with the mean value of the column.
        6. Create dummy columns for all the categorical variables in X, drop the original columns
    ***
    
    INPUT
        df: take any pandas dataframe
        dependant: define dependant as list
        drop_columns: specify if you want to drop certain columns, input as list
    
    OUTPUT
        X: A matrix holding all of the variables you want to consider when predicting the response, except the last column
        y: the corresponding response vector, always the last column of the dataframe
    """
    
    # Drop rows with missing values in dependant variable
    #y = df.iloc[:, -1:]
    #y = y.dropna(subset=y.columns.values, axis=0)
    df = df.dropna(subset=dependant, axis=0)
    y = df[dependant]
    df = df.drop(df[dependant], axis=1)
    
    #Drop certain columns, if defined
    if (drop_columns == []):
        print('No columns to drop')   
    else:
        df = df.drop(drop_columns, axis=1)
    
    # Fill numeric columns with the mean
    num_vars = df.select_dtypes(include=['float', 'int']).columns
    for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)
        
    # Dummy the categorical variables
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for var in  cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    
    X = df
    
    return X, y


# %%
#grateully taken from the couse: Udacity - Data Scientist
def coef_weights(coefficients, X_train):
    """
    ***
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    ***
    
    INPUTS:
        coefficients - the coefficients of the linear model 
        X_train - the training data, so the column names can be used
        
    OUTPUTS:
        coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    """
    
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    
    return coefs_df


# %%
def clean_fit_linear_mod(df, response_col, cat_cols, dummy_na, test_size=.3, rand_state=42):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column 
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test 
    
    OUTPUT:
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    
    Your function should:
    1. Drop the rows with missing response values
    2. Drop columns with NaN for all the values
    3. Use create_dummy_df to dummy categorical columns
    4. Fill the mean of the column for any missing values 
    5. Split your data into an X matrix and a response vector y
    6. Create training and test sets of data
    7. Instantiate a LinearRegression model with normalized data
    8. Fit your model to the training data
    9. Predict the response for the training data and the test data
    10. Obtain an rsquared value for both the training and test data
    '''
    #Drop the rows with missing response values
    df  = df.dropna(subset=[response_col], axis=0)

    #Drop columns with all NaN values
    df = df.dropna(how='all', axis=1)

    #Dummy categorical variables
    df = create_dummy_df(df, cat_cols, dummy_na)

    # Mean function
    fill_mean = lambda col: col.fillna(col.mean())
    # Fill the mean
    df = df.apply(fill_mean, axis=0)

    #Split into explanatory and response variables
    X = df.drop(response_col, axis=1)
    y = df[response_col]

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test


#Test your function with the above dataset
#test_score, train_score, lm_model, X_train, X_test, y_train, y_test = clean_fit_linear_mod(df_new, 'Salary', cat_cols_lst, dummy_na=False)


# %%



