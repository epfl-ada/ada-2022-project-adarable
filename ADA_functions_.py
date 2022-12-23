#Functions for the project "How does the COVID-19 pandemic affect Wikipedia pageviews?" by ADArable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import urllib.parse # combine URL components into URL string

import wikipediaapi # query wikipedia through api
import requests # standard for making HTTP requests

from hampel import hampel # hampel filter to detect anomalies

from pytrends.request import TrendReq # google trends
from iso3166 import countries # to derive the countries isocode2
import pickle #  to serialize and deserialize objects in Python

from datetime import timedelta

# Statistics
from statsmodels.tsa.seasonal import STL # seasonal decompositions
import statsmodels.tsa.stattools as smt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.rolling import RollingOLS
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.tsa.vector_ar.vecm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR

from scipy import signal
from scipy import stats

# interactive plots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

#import geopandas

import warnings


########################## Auxiliary functions for all research questions ########################################


def load_interventions(interventions_path='./data/interventions.csv'):
    
    """
    Loads the interventions dataset
    param: interventions_path: path to the file interventions.csv
    """

    interventions_df = pd.read_csv(interventions_path)
    for col in interventions_df.columns:
        if col != "lang":
            interventions_df.loc[:, col] = pd.to_datetime(interventions_df.loc[:, col])
    interventions = {}
    for _, lang_info in interventions_df.T.to_dict().items():
        lang = lang_info['lang']
        del lang_info['lang']
        interventions[lang] = {k: t for k, t in lang_info.items() if not pd.isnull(t)}
    return interventions


def translate(title,lan):
    
    """
    Translates the title of one Wikipedia page in another language
    param: title: title of the Wikipedia page (string)
    param: lan: desired language (string)
    return: title in the desired language (string)
    """

    lang_wiki = wikipediaapi.Wikipedia('en')
    if lan != 'en':
        page_tr =  pd.DataFrame(lang_wiki.page(title).langlinks,index = [0])
        if lan in page_tr:
            return lang_wiki.page(title).langlinks[lan].title.replace(' ','_')
    else:
        return title.replace(' ','_')


def get_article_views(article: str, location: str,start_date='20180101', end_date='20221101'):
    
    """
    Gets the weekly pageviews for one Wikipedia page in one language in the desired period
    param: article: title of the Wikipedia page
    param: location: Wikipedia language 
    param: start_date: beginning of the desired period 
    param: end_date: end of the desired period 
    return: dataframe column with the weekly pageviews
    """

    result = {}
    project = f'{location}.wikipedia.org'
    title = urllib.parse.quote(article)

    url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/all-agents/{title}/daily/{start_date}/{end_date}'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    r = requests.get(url, headers=headers)

    if list(r.json().keys())[0] != "type":
        all_views = r.json()['items']
        
        for daily_views in all_views:
            result[daily_views['timestamp']] = daily_views['views']

        result = pd.DataFrame(result,index=[article]).transpose()
        result = result.fillna(0)
        result = result.rolling(7).mean().dropna()[::7]

    return result


def get_list_wikipages(category: str, location: str) -> list:
    
    """
    Gets the list of all Wikipedia articles in one category in one language
    param: category: title of the translated Wikipedia category
    param: location: Wikipedia language  
    return: list of Wikipedia pages titles
    """

    url = f'https://{location}.wikipedia.org/w/api.php?action=query&cmlimit=max&cmtitle=Category%3A{category}&list=categorymembers&format=json'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    r = requests.get(url, headers=headers)
    
    print(url)
    all_pages = r.json()['query']['categorymembers']
    result = []
    for pages in all_pages:
        result.append(pages['title'].replace(' ', '_'))
    return result

def subpage_helper(subpage, category, pageviews, lan):

    """
    Helper function to get the weekly pageviews for one Wikipedia page in one language
    param: subpage: title of the Wikipedia page
    param: category: category of the Wikipedia page
    param: pageviews: dataframe that contains the total weekly pageviews for a list of categories
    param: lan: Wikipedia language
    """

    if len(get_article_views(subpage,lan)) == 252:
        pageviews['date'] = pd.DataFrame(get_article_views(subpage,lan)).index
        pageviews[category] = pageviews[category].values + get_article_views(subpage,lan).values

def create_dataframe_language(list_categories, lan):

    """
    Creates a dataframe that contains the total weekly pageviews for a list of categories
    param: list_categories: list of the categories of interest in English
    param: lan: Wikipedia language 
    return: dataframe with a date column and one column per category that contains the total weekly pageviews
    """

    pageviews = pd.DataFrame()

    for category in list_categories:
        if lan == 'en':
             titles_pages = pd.Series(get_list_wikipages(category, lan)).dropna().drop_duplicates()
           
        else:
            transl_cat = translate(category,lan)
            titles_pages = pd.Series(get_list_wikipages(transl_cat,lan)).dropna().drop_duplicates()
        
        #print(titles_pages)
        pageviews[category] = np.zeros(252)
                            
        titles_pages.apply(lambda x: subpage_helper(x, category, pageviews, lan))

            
    pageviews['date'] = pageviews['date'].apply(lambda x: str(x[:4])+'-'+str(x[4:6])+'-'+str(x[6:8])) 
    pageviews['date'] = pd.to_datetime(pageviews['date'], format = '%Y-%m-%d')

    return pageviews

########################## RESEARCH QUESTION 2 ########################################


def calculate_ratio(data, language, cuisine_dict):

    """
    Calculates the ratio of the pageviews of a category over the sum of the pageviews of all other categories
    param: data: dataframe with a date column and one column per category that contains the total weekly pageviews
    param: language: Wikipedia language
    param: cuisine_dict: dictionary that contains the name of the column of the category of interest in each language
    return: dataframe with a date column and one column with the ratio
    """

    ratio = pd.DataFrame()
    ratio['date'] = data['date']
    ratio['ratio'] = data.drop(columns = [cuisine_dict[language], 'date']).sum(axis = 1)/data[cuisine_dict[language]]
    return ratio


def add_stl_plot(res, dataframes, lan, category, question):

    """
    Add 3 plots from a second STL fit
    param: res: residuals obtained from the STL fit
    param: dataframes: Wikipedia pageviews
    param: lan: Wikipedia language
    param: category: name of the column of which we want to obtain the seasonal decomposition
    """
    
    colors = sns.color_palette("bright")
    fig, ax = plt.subplots(4, 1, figsize = (9,7))
    axs = fig.get_axes()
    plt.subplot(4, 1, 1)
    if question == 2:
        plt.title("Seasonal decomposition of {} in the {} Wikipedia".format(category, lan))
    if question == 3:
        plt.title("Seasonal decomposition of {} in the {} Delivery Data".format(category, lan))    
    plt.plot(dataframes[lan]['date'], dataframes[lan][category], color = colors[4], linewidth = 2)
    plt.xlabel("Date")
    plt.ylabel("Pageviews")
    #get more spacing between subplots
    plt.subplots_adjust(hspace=0.5)
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            plt.subplot(4, 1, 4)
            plt.bar(dataframes[lan]['date'], series, width=4, color =  colors[1],  linewidth = 2)
            
        elif comp == "trend":
            plt.subplot(4, 1, 2)
            
            plt.plot(dataframes[lan]['date'], series, color = colors[0],  linewidth = 2)
        else:
            plt.subplot(4, 1, 3)
            plt.plot(dataframes[lan]['date'], series,  color = colors[2],  linewidth = 2)
        plt.title(comp)
    plt.show()

def plot_pageviews(pageviews, lan, logscale = False):

    """
    Plots the pageviews of various cuisines for a specific wikipedia language (One plot)
    param: pageviews: dataframe of pageviews
    param: lan: language of the pageviews
    param: logscale: if True, the y-axis is plotted in log-scale
    """    

    colors = sns.color_palette("colorblind",len(pageviews.keys()))
    warnings.filterwarnings("ignore")
    fig, ax = plt.subplots(figsize=(16,12))
    for col,i in enumerate(pageviews):
        if i != 'date' and not logscale:
            ax.plot(pageviews['date'],pageviews[i], label = i, color = colors[col])
        if i != 'date' and logscale:
            ax.plot(pageviews['date'],np.log(pageviews[i]), label = i, color = colors[col])
    ax.legend()
    ax.set_xlabel('Date')
    if logscale: 
        ax.set_ylabel('Log pageviews')
        ax.set_title('Log pageviews of '+lan+' Wikipedia')
    else:
        ax.set_ylabel('Pageviews')
        ax.set_title('Pageviews of '+lan+' Wikipedia')
    plt.show()


def subplot_pageviews(pageviews, lan, colour = 'r', title = 'Pageviews'):
    
    """
    Plots the pageviews of various cuisines for a specific wikipedia language (One plot per cuisine)
    param: pageviews: dataframe of pageviews
    param: lan: language of the pageviews
    """

    fig, ax = plt.subplots(8,3,figsize = (20,15))
    pageviews = pageviews.reindex(sorted(pageviews.columns), axis=1)
    dates = pageviews['date']
    pageviews = pageviews.drop(columns = 'date')
    fig.suptitle('{} {}'.format(lan, title), fontsize=16)
    fig.tight_layout(pad=3.0)
    for n,category in enumerate (pageviews.columns):
        plt.subplot(8,3,n+1)
        plt.title(category)
        plt.plot(dates,pageviews[category], colour)
    
    plt.show()
    
    
def plot_highest_pageviews(pageviews, lan, topX):
    
    """
    Plots the pageviews of the topX most viewed cuisines
    param: pageviews: dataframe with pageviews
    param: lan: language of the wikipedia
    param: topX: number of top cuisines to plot
    """

    warnings.filterwarnings("ignore")
    food_sorted = pageviews.sum(axis = 0, skipna = True).sort_values(ascending=False)
    topX_pageviews = pageviews[food_sorted.index[0:topX]]
    fig, ax = plt.subplots(figsize=(10,5))
    for i in topX_pageviews:
        if i != 'date':
            ax.plot(pageviews['date'],np.log(pageviews[i]),label = i)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('log pageviews')
    ax.set_title('Top '+ str(topX) + ' pageviews of '+ lan +' Wikipedia (log scale)')
    plt.show()
    
def plot_specific_food(pageviews, lan, food, log = False):
    
    """
    Plots the pageviews of a specific food
    param: pageviews: dataframe with pageviews
    param: lan: language of the wikipedia
    param: food: food to plot
    param: logscale: if True, the y-axis is plotted in log-scale
    """

    fig, ax = plt.subplots(figsize=(10,7))
    for i in food:
        if log:
            ax.plot(pageviews['date'],np.log(pageviews[i]),label = i)
        else:
            ax.plot(pageviews['date'],pageviews[i],label = i)
    ax.legend()
    ax.set_xlabel('Date')
    if log: 
        ax.set_ylabel('Log pageviews')
        ax.set_title('Log pageviews of '+lan+' Wikipedia')
    else:
        ax.set_ylabel('Pageviews')
        ax.set_title('Pageviews of '+lan+' Wikipedia')
    plt.show
    
def plot_specific_food_normalised(pageviews, lan, food):
    
    """
    Plots the normalised and standardize pageviews of a specific food
    param: pageviews: dataframe with pageviews
    param: lan: language of the wikipedia
    param: food: food to plot
    """

    fig, ax = plt.subplots(figsize=(10,5))
    for i in food:
        ax.plot(pageviews['date'],(pageviews[i]-np.mean(pageviews[i]))/np.std(pageviews[i]),label = i)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Pageviews')
    ax.set_title('Normalised and standardised pageviews of '+lan+' Wikipedia')
    plt.show
    
def correlation_plot(pageviews, lan):
    
    """
    Plots the correlation between the different cuisines (standardized pageviews)
    param: pageviews: dataframe with pageviews of a specific wikipedia language
    param: lan: language of the wikipedia
    """

    fig, ax = plt.subplots(figsize=(20,10))
    pageviews = pageviews.reindex(sorted(pageviews.columns), axis=1)
    for element in pageviews.columns:
        pageviews[element] = (pageviews[element] - pageviews[element].mean())/pageviews[element].std()
    pageviews = pageviews.drop(columns = 'date')
    corr = pageviews.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    plt.title('Correlation plot for '+lan+' Wikipedia')
    plt.show() 
    
def get_correlation_across_wikis(pageviews, languages):
    """
    Plots the correlation between the different cuisines of different wikipedia languages (standardized pageviews)
    param: pageviews: dataframe with pageviews of a specific wikipedia language
    param: languages: language of the wikipedia
    """
    warnings.filterwarnings("ignore")    
    totaldata = pd.DataFrame()
    for element in languages:
        totaldata[element] = pageviews[element].sum(axis = 1, skipna = True)
        totaldata[element] = (totaldata[element] - totaldata[element].mean())/totaldata[element].std()
    corr = totaldata.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    plt.title('Correlation plot of all cuisines across all Wikipedia languages')
    plt.show()
    
def get_correlation_across_wikis_for_specific_food(pageviews, languages, food):
    
    """
    Plots the correlation between the different cuisines of different wikipedia languages (standardized pageviews)
    param: pageviews: dataframe with pageviews of a specific wikipedia language
    param: languages: language of the wikipedia
    param: food: food to plot
    """

    totaldata = pd.DataFrame()
    for element in languages:
        totaldata[element] = pageviews[element][food]
        totaldata[element] = (totaldata[element] - totaldata[element].mean())/totaldata[element].std()
    corr = totaldata.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    plt.title('Correlation plot for '+ food +' across all Wikipedia language')
    plt.show()
    
def intersection(lst1, lst2):
    
    """
    Finds the intersection of two lists
    param: lst1: list 1
    param: lst2: list 2
    """

    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def remove_outliers_plot(pageviews, language, category, window_size=100, std_dev=10, question = 3):
    
    """
    Remove outliers from the data
    param: pageviews: dataframe with the pageviews
    param: language: language of the wikipedia
    param: category: category
    param: window_size: size of the window for the rolling mean
    param: std_dev: number of standard deviations to consider as outliers
    """
   
    fig, ax = plt.subplots(figsize=(10,5))
    ts_imputation = hampel(pageviews[language][category], window_size=window_size, n=std_dev, imputation=True)
    pageviews[language][category].plot(style="r-")
    ts_imputation.plot(style="g-")
    plt.xlabel('Date')
    plt.ylabel('Pageviews')
    if question == 2:
        plt.title("Seasonal decomposition of {} in the {} Wikipedia".format(category, language))
    if question == 3:
        plt.title("Seasonal decomposition of {} in the {} Delivery Data".format(category, language))
    plt.show()
    return ts_imputation

def remove_outliers(pageviews, language, window_size=10, std_dev=10):
    
    """
    Remove outliers from the data
    param: pageviews: dataframe with the pageviews
    param: language: language of the wikipedia
    param: food: food category
    param: window_size: size of the window for the rolling mean
    param: std_dev: number of standard deviations to consider as outliers
    """

    for food in pageviews[language].columns:
        if food != 'date':
            ts_imputation = hampel(pageviews[language][food], window_size=window_size, n=std_dev, imputation=True)
            pageviews[language][food] = ts_imputation
    return pageviews[language]

def add_stl_plot(res, dataframes, lan, category, question):
    
    """
    Add 3 plots from a second STL fit
    param: res: residuals obtained from the STL fit
    param: dataframes: Wikipedia pageviews
    param: lan: Wikipedia language
    param: category: name of the column of which we want to obtain the seasonal decomposition
    """
    
    colors = sns.color_palette("bright")
    fig, ax = plt.subplots(4, 1, figsize = (9,7))
    axs = fig.get_axes()
    plt.subplot(4, 1, 1)
    if question == 2:
        plt.title("Seasonal decomposition of {} in the {} Wikipedia".format(category, lan))
    if question == 3:
        plt.title("Seasonal decomposition of {} in the {} Delivery Data".format(category, lan))
        
    plt.plot(dataframes[lan]['date'], dataframes[lan][category], color = colors[4], linewidth = 2)
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            plt.subplot(4, 1, 4)
            plt.bar(dataframes[lan]['date'], series, width=4, color =  colors[1],  linewidth = 2)
            
        elif comp == "trend":
            plt.subplot(4, 1, 2)
            
            plt.plot(dataframes[lan]['date'], series, color = colors[0],  linewidth = 2)
        else:
            plt.subplot(4, 1, 3)
            plt.plot(dataframes[lan]['date'], series,  color = colors[2],  linewidth = 2)
        plt.title(comp)

def remove_seasonality(data, seasonalitytype = 'Trend', period = 52, robust = True):
    
    """
    Remove seasonality from the data
    param: data: Wikipedia pageviews
    param: seasonalitytype: defines whether we keep only the trend (Trend), the residuals (Res), both (Both) or all (All (including the seasonal trend)) of the seasonality decomposition
    param: period: Period of the seasonality
    param: robust: Adaptation for more robustnes
    """

    if seasonalitytype == 'All':
        return data
    removed_seasonality = pd.DataFrame()
    removed_seasonality['date'] = data['date']
    data = data.drop('date', axis = 1)
    for food in data.columns:
        if seasonalitytype == 'Both':
            removed_seasonality[food]  =  getattr(STL(data[food], period=52, robust=True).fit(),'trend') + getattr(STL(data[food], period=52, robust=True).fit(),'resid')
        if seasonalitytype == 'Trend':
            removed_seasonality[food]  = getattr(STL(data[food], period=52, robust=True).fit(),'trend')
        if seasonalitytype == 'Res':
            removed_seasonality[food] = getattr(STL(data[food], period=52, robust=True).fit(),'resid')
    return removed_seasonality

def data_cleaning(pageviews, language, seasonalitytype = 'Trend',period = 10, robust = True):
    
    """
    Function to clean the data by removing outliers and seasonality
    param: pageviews: dictionary of dataframes with pageviews for each language
    param: language: language of interest
    param: period: period of seasonality to remove
    param: robust: boolean to indicate whether to use robust method to remove seasonality
    """

    data_outliers_removed = remove_outliers(pageviews, language)
    data_removed_seasonality = remove_seasonality(data_outliers_removed, seasonalitytype, period, robust)
    return data_removed_seasonality

def standardize(data):
    
    """
    Function to standardize the data (z-score)
    param: data: dataframe to standardize
    """

    for col in data.columns:
        if col != 'date':
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data

def minmax(data):
    
    """
    Function to standardize the data (max-min)
    param: data: dataframe to standardize
    """

    for col in data.columns:
        if col != 'date':
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data

def pipeline_calculate_ratios(pageviews, wikipedia_languages, cuisine_dict, colour, seasonalitytype = 'Trend', period = 10, robust = True):
    
    """
    Function to calculate the ratio between pageviews of the cuisine corresponding to the wikipedia language and all other cuisines
    param: pageviews: dictionary of dataframes with pageviews for each language
    param: wikipedia_languages: list of wikipedia languages
    param: cuisine_dict: dictionary with the cuisine corresponding to each wikipedia language
    param: colour: list of colours to use for the plot
    param: period: period of seasonality to remove
    param: robust: boolean to indicate whether to use robust method to remove seasonality
    """

    ratios = {}
    fig, ax = plt.subplots(8, 1, figsize = (5,20))
    axs = fig.get_axes()
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=5.0)
    for n, lan in enumerate (wikipedia_languages):
        data_removed_seasonality = data_cleaning(pageviews, lan, seasonalitytype, period, robust)
        ratios[lan] = calculate_ratio(data_removed_seasonality, lan, cuisine_dict)
        plt.subplot(8, 1, n+1)
        plt.plot(ratios[lan]['date'], ratios[lan]['ratio'], c = colour[n])
        plt.title('Ratio between pageviews of {} cuisine and all other cuisines'.format(lan))
        plt.xlabel('Date')
        plt.ylabel('Ratio')
    plt.show()  
    return ratios


iso = ['ITA', 'USA', 'FRA', 'NOR', 'DEU', 'AUT', 'GBR', 'CHN','THA', 'JPN', 'ESP', 'MEX', 'IND', 'TUR', 'GRC', 'KOR','VNM', 
        'FIN', 'DNK', 'SRB', 'ZAF','NLD', 'SWE', 'ARE']
continent = ['Europe', 'America', 'Europe', 'Europe', 'Europe', 'Europe', 'Europe', 'Asia', 'Asia', 'Asia', 'Europe', 
        'America', 'Asia', 'Europe', 'Europe', 'Asia', 'Asia', 'Europe', 'Europe', 'Europe', 'Africa','Europe', 'Europe', 'Asia']
popular = ['Risotto, Pizza, Pasta', 'Hamburger, Barbecue, Mac n Cheese', 'Cassoulet, Baguette, Coq au vin', 
        'Fårikål, Lobscouse, Frikadeller', 'Cabbage, Pretzel, Wurstel', 'Schnitzel, Apfelstrudl', 
        'Fish & Chips, Mash, Roast', 'Dumplings, Spring rolls, Cantonese rice', 'Pad thai, Tom yum soup, Som tam salad', 
        'Sushi, Tempura, Ramen', 'Gazpacho, Paella, Tortilla', 'Chilaquiles, Pozole, Tacos','Curry, Tandoori chicken, Samosa', 
        'Baklava, Kebap, Doner', 'Moussaka, Gyros, Papoutsakia', 'Kimchi, Bibimap, Bulgogi', 'Pho soup, Bun cha, Bánh mì', 
        'Karjalanpiirakka, Ruisleipa, Leipajuusto ', 'Smørrebrød, Stegt Flæsk, Kanelsnegl', 'Sarma, Ćevapi, Burek', 'Bobotie, Biltong, Vetkoek',
        'Poffertjes, Pannenkoeken, Erwtensoep ', 'Köttbullar, Räkmacka, Smulpaj', 'Hummus, Falafel, Manousheh']    


def create_dataframe_countries(dataframe, iso = iso , continent = continent, popular = popular):
    
    """
    Create a dataframe with the weekly pageviews for each country
    input: dataframe = dataframe with the weekly pageviews for each language
    input: iso = list of iso codes for each country
    input: continent = list of continents for each country
    input: popular = list of popular cuisines for each country
    """

    countries = dataframe.columns[:-1]
    df = pd.DataFrame()
    counter = 0
    df['country'] = 'NA'
    df['iso'] = 'NA'
    df['continent'] = 'NA'
    df['popular'] = 'NA'
    for country in countries:
        df = pd.concat([df, dataframe[country]], axis = 0)
        df['country'].iloc[counter*253:(counter+1)*253] = country
        df['iso'].iloc[counter*253:(counter+1)*253] = iso[counter]
        df['continent'].iloc[counter*253:(counter+1)*253] = continent[counter]
        df['popular'].iloc[counter*253:(counter+1)*253] = popular[counter]
        counter += 1
    df['Date'] = dataframe['Date']
    df = df.reset_index()
    df = df.rename(columns = {'index':'date', 0:'pageviews'})
    df = df.drop(columns = 'date')
    return df

def get_interactive_plots(pageviews, lan, seasonalitytype = 'Trend', standardizationtype = 'minmax', plottype = 'landscape', period = 10, robust = True):
    
    """
    Create an interactive plot with the weekly pageviews for each language
    input: pageviews = dataframe with the weekly pageviews for each language
    input: lan = wikipedia language
    input: standardizationtype = type of standardization to be applied to the data (std or minmax)
    input: plottype = type of plot to be created (landscape or globe)
    input: period = period of the seasonality to be removed
    input: robust = boolean to decide if the seasonality should be removed using a robust method
    output: interactive plot
    """

    # ignore warnings
    warnings.filterwarnings("ignore")
    data_removed_seasonality = data_cleaning(pageviews, lan, seasonalitytype, period, robust)
    data_removed_seasonality_adapted = data_removed_seasonality.reset_index()
    data_removed_seasonality_adapted = data_removed_seasonality_adapted.drop(columns = ['index'])
    if standardizationtype == 'std':
        data_removed_seasonality_adapted = standardize(data_removed_seasonality)
        print("ADAPT RANGE COLOUR!")
    elif standardizationtype == 'minmax':
        data_removed_seasonality_adapted = minmax(data_removed_seasonality)
    data_removed_seasonality['date'] = data_removed_seasonality['date'].astype(str)
    data_removed_seasonality_adapted['Date'] = data_removed_seasonality['date']
    data_removed_seasonality_adapted = data_removed_seasonality_adapted.drop(columns =  'date')
    data_toplot = create_dataframe_countries(data_removed_seasonality_adapted)
    if plottype == 'landscape':
        #set title for figure   
        fig = px.choropleth(data_toplot, locations='iso', color='pageviews', hover_name='country', hover_data = ['continent', 'popular'], animation_frame='Date', range_color=(0, 1), title = "Evolution of standardized pageviews for world cuisines for the {} Wikipedia".format(lan), )
        fig.show()  
        pio.write_html(fig,"WorldMap1.html")
    elif plottype == 'globe':
        map_fig = px.scatter_geo(data_toplot, 
                         locations = 'iso',
                         projection = 'orthographic',
                         color = 'continent',
                         opacity = 1,
                         hover_name = 'country',
                         hover_data = ['popular', 'continent']
                        )

        map_fig.show()
  
def get_mobility_data_together_with_pageviews(mob_data_lan, ratios, interventions, languages, colours):
    
    """
    Plot the mobility data together with the pageviews ratio and the interventions
    input: mob_data_lan = dictionary with the mobility data for each language
    input: ratios = dictionary with the pageviews ratio for each language
    input: interventions = dictionary with the interventions for each language
    input: languages = list of languages
    input: colours = list of colours
    output: plot
    """

    fig, ax = plt.subplots(7, 1, figsize = (5,20))
    axs = fig.get_axes()
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=5.0)
    for n, lan in enumerate(languages):
        plt.subplot(7, 1, n+1)
        plt.title("{} wikipedia: ratio of {} cuisine vs other cuisine pageviews, mobility data and interventions".format(lan, lan))
        plt.plot(ratios[lan]['date'], ratios[lan]['ratio'], color = colours[2*n], linewidth = 2, label = 'Pageviews ratio')
        ax2 = plt.twinx()
        ax2.plot(mob_data_lan[lan].index, mob_data_lan[lan]['residential_percent_change_from_baseline'], color = colours[2*n+1], label = 'Mobility data')
        plt.axvline(x = pd.Series(interventions[lan]['Mobility']), linewidth=2, color = colours[-1], linestyle = '--', label = 'First intervention')
    plt.show()



def plot_category(dataframe, mob_data_lan, interventions, languages, method, question):
    
    """
    Plots the food delivery usage in the different countries
    
    input: dataframe = dict of pd.DataFrames containing time series data to be plotted
           mob_data_lan = dict of pd.DataFrames containing Google mobility data to be plotted 
           interventions = dict containing important COVID dates (e.g., Lockdown, Mobility, Normality)
           languages = list of the languages under study
           method = ['raw', 'Trend', TrendResid', 'Resid']
           question = [2,3] depending on the research question 
    """
    
    
    fig, ax = plt.subplots(4, 2, figsize = (17,15))
    ax = ax.flatten()
    col = sns.color_palette("colorblind", 4)
    fig.tight_layout(pad=5.0)
    if question == 2:
        fig.suptitle('Food cultures', fontsize=16)
        
        for n,lan in enumerate(languages):
            plt.subplot(4, 2, n+1)
            plt.title(lan)
            first = pd.Series(interventions[lan]['Mobility'])
            if method == 'All':
                plt.plot(dataframe[lan]['date'],dataframe[lan]['ratio'], label = 'Ratio food cultures')
            if method == 'Both':
                 plt.plot(dataframe[lan]['date'],dataframe[lan]['ratio'], label = 'Ratio food cultures')
            if method == 'Trend':
                 plt.plot(dataframe[lan]['date'],dataframe[lan]['ratio'], label = 'Ratio food cultures')
            if method == 'Res':
                 plt.plot(dataframe[lan]['date'],dataframe[lan]['ratio'], label = 'Ratio food cultures')

            ax2=ax[n].twinx()
            ax2.plot(mob_data_lan[lan]['date'],mob_data_lan[lan]['residential_percent_change_from_baseline'].rolling(1).mean(), color = col[1], ls = '-',  label = 'Mobility data')
            #ax2.plot(mob_data_lan[lan]['date'],mob_data_lan[lan]['retail_and_recreation_percent_change_from_baseline'].rolling(3).mean(), color = col[1], ls = '-',  label = 'Mobility data')
            date = dataframe[lan]['date']
            plt.axvline(x=first, linewidth=2, color='violet', label = 'Lockdown')
            plt.axvline(x=pd.Series(interventions[lan]['Second_Normalcy']), linewidth=2, color=col[2], label = 'Second Normalcy')
            plt.axvline(x= pd.Series(interventions[lan]['Normalcy']), linewidth=2, color=col[3], label = 'First Normalcy')
            
        
    if question == 3:
        fig.suptitle('Food delivery services: ' + method, fontsize=16)

        for n,lan in enumerate(languages):
            plt.subplot(4, 2, n+1)
            plt.title(lan)
            first = pd.Series(interventions[lan]['Mobility'])
            
            plt.plot(dataframe[lan]['date'],dataframe[lan]['sum_norm_' + method], label = 'Google searches')
            ax2=ax[n].twinx()
            ax2.plot(mob_data_lan[lan]['date'],mob_data_lan[lan]['retail_and_recreation_percent_change_from_baseline'].rolling(3).mean(), color = col[1], ls = '-',  label = 'Mobility data')
            date = dataframe[lan]['date']

            plt.axvline(x=first, linewidth=2, color='violet', label = 'Lockdown')
            plt.axvline(x=pd.Series(interventions[lan]['Second_Normalcy']), linewidth=2, color=col[2], label = 'Second Normalcy')
            plt.axvline(x= pd.Series(interventions[lan]['Normalcy']), linewidth=2, color=col[3], label = 'First Normalcy')

        plt.legend(loc = 'upper left')
        
def independency(time_series, lan, method, interventions, question):
    
    """
    checking the the null hypothesis that the time series before and after COVID have identical average (expected) values.
    input: time_series = pd.Series, time series under analysis 
           lan = str, language under study
           method = ['raw', 'Trend', TrendResid', 'Resid']
           interventions = dict, containing important COVID dates (e.g., Lockdown, Mobility, Normality)
           question: [2,3] depending on the research question 
    return the results of the independent t-test
    """

    if question == 2:
    
        if method == 'All':
            time_series_pre = time_series[lan][time_series[lan]['date'] < interventions[lan]['Mobility']]['ratio'].dropna()
            time_series_post = time_series[lan][time_series[lan]['date'] > interventions[lan]['Mobility']]['ratio']
        if method == 'Both':
            time_series_pre = time_series[lan][time_series[lan]['date'] < interventions[lan]['Mobility']]['ratio'].dropna()
            time_series_post = time_series[lan][time_series[lan]['date'] > interventions[lan]['Mobility']]['ratio']
        if method == 'Trend':
            time_series_pre = time_series[lan][time_series[lan]['date'] < interventions[lan]['Mobility']]['ratio'].dropna()
            time_series_post = time_series[lan][time_series[lan]['date'] > interventions[lan]['Mobility']]['ratio']
        if method == 'Res':
            time_series_pre = time_series[lan][time_series[lan]['date'] < interventions[lan]['Mobility']]['ratio'].dropna()
            time_series_post = time_series[lan][time_series[lan]['date'] > interventions[lan]['Mobility']]['ratio']
            
    if question == 3:
 
        time_series_pre = time_series[lan][time_series[lan]['date'] <interventions[lan]['Mobility']]['sum_norm_' + method].dropna()
        time_series_post = time_series[lan][time_series[lan]['date'] > interventions[lan]['Mobility']]['sum_norm_' + method]

    result = stats.ttest_ind(time_series_pre,time_series_post, equal_var= False, alternative = 'less')
    print('Language: {0}, t-statistics: {1}, p-value: {2}'.format(lan, result[0], result[1]))
    return 



########################## RESEARCH QUESTION 3 ########################################

def create_delivery_dataset(searches, successful_languages):

    """
    Creates a dataframe that contains the total Google searches for a list of categories
    
    input: searches: list of the categories of interest 
    return: dataframe with a date column, one column per category that contains the total weekly Google searches, a column with the sum of all the categories
    """
    
    food_delivery = pd.DataFrame()
    food_delivery_countries = {}
    for num,country in enumerate(isocode2[3:5]):
        food_delivery = pd.DataFrame()
        for company in searches:
            #pytrends.build_payload([company], timeframe='2018-01-01 2022-11-01', geo = country ,gprop='')
            pytrends.build_payload([company], timeframe='2018-01-01 2022-11-01', geo = country ,gprop='')
            data = pytrends.interest_over_time()

            if not data.empty:
                food_delivery[company] = hampel(data.reset_index()[company],window_size=10, n=2, imputation=True)
                food_delivery['date'] = data.reset_index()['date']

        food_delivery['sum'] = food_delivery.sum(numeric_only = True,axis=1)
        food_delivery_countries[successful_languages[num+3]] = food_delivery
    return food_delivery_countries

def granger_causality(food_delivery,mobility, lan, lags = 4):
    
    """
    verifying the causality of mobility data on food delivery services preferences
    input: food_delivery = dict of pd.DataFrames containing food delivery service data
           mobility = dict of pd.DataFrames containing Google mobility data to be plotted 
           lan = language under study
           lags = maximum number of weeks we want to check the causality for
    return the results of the granger causality
    
    """

    if len(food_delivery) <= 14:
        lags = 1
    if len(food_delivery) <= 16 and len(food_delivery)>14:
        lags = 2
    time_series = pd.DataFrame({'food_delivery':food_delivery.values,'mobility':mobility.values})
    ad = [adfuller(time_series.iloc[:,0]),adfuller(time_series.iloc[:,1])]
    while ((ad[0][1] > 0.05) | (ad[1][1] > 0.05)): # this looks if the time series is stationary
        time_series = time_series.diff().dropna()
        ad = [adfuller(time_series.iloc[:,0]),adfuller(time_series.iloc[:,1])]
            
    result = grangercausalitytests(time_series, maxlag=lags,verbose= False);
    p_value = 1
    for lag in result:
        if result[lag][0]['ssr_ftest'][1] < 0.05:
            p_value = result[lag][0]['ssr_ftest'][1]
            week = lag
            print('Granger causality for language {0}: p_value at week {1} = {2}'.format(lan,week,p_value)) 
            print(' ')
            return p_value, week
        elif result[lag][0]['ssr_ftest'][1] < p_value:
            p_value = result[lag][0]['ssr_ftest'][1]
            week = lag
    print('Granger causality for language {0}: p_value at week {1} = {2}'.format(lan,week,p_value)) 
    print(' ')
    return p_value, week

def pre_proc(data, method):
    
    """
    Filter the time series with an hampel filter, window size = 5, threshold (std) = 3 
    Depending on the input method, the function decompose the time series in raw, trend+resid, trend, resid
    
    
    input: data = pd.Series we want to pre-process
           method = ['raw','trend_resid', 'trend', 'resid']
           
    return: the pre processed time series
    """

    y_noout = hampel(data,window_size=5, n=3, imputation=True)
    
    if method == 'raw':
        y = y_noout
    if method == 'trend_resid':
        y =  getattr(STL(y_noout, period=52, robust=True).fit(),'trend') + getattr(STL(y_noout, period=52, robust=True).fit(),'resid')
    if method == 'trend':
        y = getattr(STL(y_noout, period=52, robust=True).fit(),'trend')
    if method == 'resid':
        y = getattr(STL(y_noout, period=52, robust=True).fit(),'resid')
    return y.rolling(3,min_periods = 1).mean()   


def rolling_regression(a,b):

    """
    Rolling regression of a on b, with a window of 4 weeks
    input: a = pd.Series
    input: b = pd.Series
    return: a dataframe with the estimated coefficient, t-value and 95% confidence interval of b
    """

    a = preprocessing.scale(a)
    b = preprocessing.scale(b)
    df = pd.DataFrame({'a':a, 'b':b})

    model = RollingOLS.from_formula('a ~ b -1', data = df, window=4)

    reg_obj = model.fit()
    
    reg_rls = sm.RecursiveLS.from_formula('a ~ b -1', df)
    model_rls = reg_rls.fit()

    # estimated coefficient
    b_coeff = reg_obj.params['b'].rename('coef')

    # b t-value 
    b_t_val = reg_obj.tvalues['b'].rename('t')

    # 95 % confidence interval of b
    b_conf_int = reg_obj.conf_int(cols=[0]).droplevel(level=0, axis=1)

    # join all the desired information to the original df
    df = df.join([b_coeff, b_t_val, b_conf_int])
    return reg_obj, model_rls, reg_rls


def cointegration_test(df, alpha=0.05): 
    
    """
    Perform Johanson's Cointegration Test and Report Summary
    input: df = pd.DataFrame you want to run the test on
           alpha = significance level
    
    """
    
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        
def stationariety(time_series):
    
    """
    Checking the stationariety of time series and deriving them to make them stationary when needed
    
    input: pd.DataFrame containing the time series to make stationary
    
    return the stationary time series and their order of integration
    """
    
    d = 0
    ad = [adfuller(time_series.iloc[:,0]),adfuller(time_series.iloc[:,1])]
    while ((ad[0][1] > 0.05) | (ad[1][1] > 0.05)): # this looks if the time series is stationary
        time_series = time_series.diff(axis = 0).dropna()
        ad =  [adfuller(time_series.iloc[:,0]),adfuller(time_series.iloc[:,1])]
        d = d+1
    #print(ad[0][1] > 0.05,ad[1][1] > 0.05, i)
    return time_series, d


def adjust(val, length= 6): return str(val).ljust(length)


def forecast(model, df_diff, df, lan, d, nobs):
    
    """
    Applying the forecasting
    
    input: model = fitted VAR model 
           df_diff = pd.DataFrame containing the stationary time series to use as input for the forecasting
           df = pd.DataFrame, initial dataset on which to retrieve the index of the forecasted dataframe
           lan = language under study
           nobs = number of observations to forecast
           d = order of integration
    return: df.DataFrame with nobs forecasted observations
    """

    lag_order = model[lan].k_ar
    forecast_input = df_diff[lan].values[-lag_order:]
    fc = model[lan].forecast(y=forecast_input, steps = nobs)
    if d == 0:
        df_forecast = pd.DataFrame(fc, index=df[lan].index[-nobs:], columns=df[lan].columns+'_forecast')        
    if d == 1:
         df_forecast = pd.DataFrame(fc, index=df[lan].index[-nobs:], columns=df[lan].columns + '_1d')
    if d == 2:
        df_forecast = pd.DataFrame(fc, index=df[lan].index[-nobs:], columns=df[lan].columns + '_2d')
        
    return df_forecast
    
def invert_transformation(df_train, df_forecast,  d):
    
    """
    Revert back the differencing to get the forecast to original scale.
    
    input: df_train = pd.DataFrame containing the time series used for the training of the forecasting
           df_forecast = pd.DataFrame containing the forecasted observations
           d = order of integration
    
    return: pd.DataFrame containing the forecasted observations at the original scale           
    """

    if d != 0:
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:  
            if d == 1:      
            # Roll back 1st Diff
                df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
            if d == 2:
                df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
            # Roll back 1st Diff
                df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    else:
        df_fc = df_forecast.copy()       
        
    return df_fc   


def forecast_accuracy(forecast, actual):
    
    """
    returning the mean percentage error and the Pearson product-moment correlation coefficients  of the forecasting
    input: forecast = forecasted values
            actual = test set values
    """
    
    mpe = np.mean((forecast - actual)/actual)   # MPE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    return({'mpe': mpe,'corr':corr})



titles = ['Italian (it)', 'German (de)', 'Dutch (nl)', 'Serbian (sr)', 'Turkish (tr)', 'Catalan (ca)', 'Finnish (fi)', 'English (en)']

def create_layout_button(k, customer, Lc):

    """
    Create a button for the interactive plot
    input: k = index of the button
            customer = button options
            Lc = number options
    return: dictionary with the button options
    """

    visibility= [False]*5*Lc
    for tr in np.arange(5*k, 5*k+5):
        visibility[tr] =True
    return dict(label = customer, method = 'update', args = [{'visible': visibility,'title': customer,'showlegend': True}])   

def get_interactive_pageviews(data, titles, wiki_languages, log = False):
    
    """
    Create an interactive plot with the pageviews of the different cuisines for the selected wikipedia language
    input: data = pd.DataFrame containing the pageviews
            titles = list of the titles of the different cuisines
            wiki_languages = list of the wikipedia languages
            log = boolean, if True the pageviews are plotted in log scale
    return: plotly figure
    """

    buttons = []
    fig = make_subplots(y_title = 'log (Pageviews)')
    fig.add_annotation(text="Distribution of pageviews of different cuisines for the selected wikipedia language", xref="paper", yref="paper", x=0.5, y=1.1, showarrow=False, font_size=20)
    col = sns.color_palette("colorblind", 4)
    country_list = wiki_languages
    for country in country_list:
        for cuisine in data[country].columns:
            if cuisine == 'date':
                continue
            if log and not data[country][cuisine] == 0:
                fig.add_trace(go.Box(y = np.log(data[country][cuisine]),name = cuisine + ' ' +country))
            else:
                fig.add_trace(go.Box(y = data[country][cuisine],name = cuisine + ' ' +country))
        
    Ld=len(fig.data)
    Lc =len(wiki_languages)
    for k in range(24, Ld):
        fig.update_traces(visible=False, selector = k)
 

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = [create_layout_button(k, customer) for k, customer in enumerate(titles)]
            )
        ]) 
    fig.show()

# Helper functions specific to question 1

def count_pages_en(category):
    
    """
    Counts the number of pages in the category in the English language
    param: category = name of category
    return: number of pages in the category
    """
    
    titles_pages_en = pd.Series(get_list_wikipages(category, 'en')).dropna().drop_duplicates()
    return len(titles_pages_en)

def create_dataframe_language2(list_categories, lan):
    
    """
    Creates a dataframe that contains the total weekly pageviews for a list of categories
    param: list_categories: list of the categories of interest in English
    param: lan: Wikipedia language 
    return: dataframe with a date column and one column per category that contains the total weekly pageviews
    """
    
    pageviews = pd.DataFrame()

    for category in list_categories:
        titles_pages_en = pd.Series(get_list_wikipages(category, 'en')).dropna().drop_duplicates()
        if lan != 'en':
            titles_pages = pd.Series([translate(i,lan) for i in titles_pages_en]).dropna().drop_duplicates() # we get the translated titles
        else:
            titles_pages = titles_pages_en
        
        pageviews[category] = np.zeros(252)                    
        titles_pages.apply(lambda x: subpage_helper(x, category, pageviews, lan))

            
    pageviews['date'] = pageviews['date'].apply(lambda x: str(x[:4])+'-'+str(x[4:6])+'-'+str(x[6:8])) 
    pageviews['date'] = pd.to_datetime(pageviews['date'], format = '%Y-%m-%d')

    return pageviews

def remove_outliers2(pageviews, language, food, window_size, std_dev):
    """
    Remove outliers from the data
    param: pageviews: dataframe with the pageviews
    param: language: language of the wikipedia
    param: food: food category
    param: window_size: size of the window for the rolling mean
    param: std_dev: number of standard deviations to consider as outliers
    return: pageviews time series without outliers
    """
    ts_imputation = hampel(pageviews[language][food], window_size, std_dev, imputation=True)
    return ts_imputation

def z_score(x, lan, health, month, categories):
    
    """
    Perform z-score standardization on pageviews with pageviews from the same month of 2019
    param: x: row of dataframe with the pageviews during covid
    param: lan: language of the wikipedia
    param: health: boolean indicating healthy or unhealthy food data
    param: month: month (as an integer) of the pageviews data to be standardized
    param: categories: list of names of food categories
    return: standardized row of dataframe with the pageviews
    """
    
    if health == 1:
        means = healthy_means[lan].iloc[month-1]
        stds = healthy_stds[lan].iloc[month-1]
    else:
        means = unhealthy_means[lan].iloc[month-1]
        stds = unhealthy_stds[lan].iloc[month-1]
    
    new_row = {}
    for food in categories:
        new_value = x[food] - means[food]
        new_value = new_value/stds[food]
        if math.isnan(new_value): # Avoid division by zero problem: replace nan by 0
            new_row[food] = 0
        else:
            new_row[food] = new_value
    return new_row


def interactive_mob_plot(df, df2, health, column, title_):
    
    """
    Make an interactive plot of mobility data along with pageviews
    param: df: dataframe containing pageviews data
    param: df2: dataframe with the mobility data
    param: health: string, healthy or unhealthy
    param: column: column with the pageviews data we are plotting with the mobility data
    param: title_ : title given to the graph
    return: interactive plot of pageviews and mobility data
    """

    buttons = []
    i = 0
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    col = sns.color_palette("colorblind", 4)
    country_list = successful_languages
    for country in country_list:
        fig.add_trace(go.Scatter(x = df2[country]['date'],y = df2[country]['residential_percent_change_from_baseline'],name = 'Mobility data' + ' ' +country, fillcolor = 'yellow'), secondary_y=True )
        fig.add_trace(go.Scatter(x = df[country]['date'], y = df[country][column],name = health +' pageviews' + ' ' +country, visible = (i==0), fillcolor = 'aqua'))
        fig.add_trace(go.Scatter(x = [interventions[country]['Normalcy'],interventions[country]['Normalcy']], y= [np.min(df[country][column]),np.max(df[country][column])], mode='lines',name='Normalcy ', visible = (i==0)))
        fig.add_trace(go.Scatter(x = [interventions[country]['Mobility'],interventions[country]['Mobility']], y= [np.min(df[country][column]),np.max(df[country][column])], mode='lines',name='Lockdown ', visible = (i==0), fillcolor = 'moccasin'))
        fig.add_trace(go.Scatter(x = [interventions[country]['Second_Normalcy'],interventions[country]['Second_Normalcy']], y= [np.min(df[country][column]),np.max(df[country][column])], mode='lines',name='Second Normalcy ', visible = (i==0)))



    Ld=len(fig.data)
    Lc =len(successful_languages)
    for k in range(5, Ld):
        fig.update_traces(visible=False, selector = k)
    
    fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active = 0,
        buttons = [create_layout_button(k, customer) for k, customer in enumerate(successful_languages)],
        x = 1,
        xanchor = 'left',
        y = 1.2,
        yanchor = 'top',
        )
    ])


    fig.update_layout(

        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),)


    fig.update_layout(
        autosize=False,
        width=1100,
         height=500)
    
    
    fig.update_layout(
    title={
        'text': title_,
        'y':0.9,
        'x':0.4,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_yaxes(title_text="<b>"+health +" pageviews"+"</b> data", secondary_y=False)
    fig.update_yaxes(title_text="<b>Mobility</b> data", secondary_y=True)
    
    pio.write_html(fig, health +" pageviews_mob.html")
    fig.show()
    
def get_spearmanr(mob, ratio, lan, event, period):
    
    """
    Get Spearman's correlation coefficient between the ratio and mobility data, around an event of the covid timeline
    param: mob: dataframe containing mobility data
    param: ratio: dataframe with the ratio data
    param: lan: language of Wikipedia considered
    param: event: date of an event in the pandemic timeline
    param: period: number of days to be considered around the event (in a symmetric manner)
    return: Spearman's correlation coefficient
    return: p-value
    """

    if period:
        x = mob[lan][(mob[lan]['date'] >= event - timedelta(days=period)) & (mob[lan]['date'] <= event+timedelta(days=period))]['residential_percent_change_from_baseline']
        y = ratio[lan][(ratio[lan]['date'] >= event - timedelta(days=period))&(ratio[lan]['date'] <= event+timedelta(days=period))]['ratio']
    else:
        x = mob[lan]['residential_percent_change_from_baseline']
        y = ratio[lan]['ratio']
    [corr, p] = scipy.stats.spearmanr(x, y)
    return corr, p
    
def plot_spearman(mob_data_lan, ratio, lan, event, period):
    
    """
    Plot a boxplot with the Spearman's correlation coefficient for different languages
    param: mob_data_lan: dataframe containing mobility data
    param: ratio: dataframe with the ratio data
    param: lan: language of Wikipedia considered
    param: event: name of an event in the pandemic timeline
    param: period: number of days to be considered around the event (in a symmetric manner)
    return: Boxplot of Spearman's correlation coefficient
    """    
    
    plt.rcParams.update({'font.size': 30})
    
    spearman_list = []
    for lan in successful_languages:
        entry = {'language': lan}
        if event:
            [corr, p] = get_spearmanr(mob_data_lan, ratio, lan, interventions[lan][event], period)
        else:
            [corr, p] = get_spearmanr(mob_data_lan, ratio, lan, None, None)
        entry['corr'] = corr
        entry['pvalue'] = p
        spearman_list.append(entry)
    df_plot = pd.DataFrame(spearman_list)
    df_plot['p<0.05'] = df_plot['pvalue'].apply(lambda x: 'p<0.05' if x <0.05 else 'p>=0.05')
    
    f, ax = plt.subplots(figsize=(35,6))
    sns.boxplot(x="corr", data=df_plot, orient = 'h',  width = 40, color = 'lavender')
    sc1 = Line2D([0], [0], marker='o', color='coral', label='p>=0.05', linestyle = 'None', markerfacecolor='coral', markersize=10)
    sc2 = Line2D([0], [0], marker='o', color='xkcd:kiwi green', label='p<0.05', linestyle = 'None', markerfacecolor='xkcd:kiwi green', markersize=10)

    
    sns.stripplot(x="corr", hue = 'p<0.05',data=df_plot, orient = 'h', size = 20, marker = 'o',edgecolor='gray',alpha = 0.7,palette=['xkcd:kiwi green','coral'])
    plt.vlines(0,-30,30, linestyle = '--',color = 'black')
    
    for lan in df_plot['language'].unique():
        sns.stripplot(x="corr", hue = 'p<0.05',data=df_plot[df_plot['language'] == lan], orient = 'h', jitter =1.2, size = 20, marker = '$'+lan+'$',palette=['black','black'])
    
    plt.legend(handles=[sc1, sc2], loc = 'lower right', fontsize=30)
    plt.xlabel("Spearman's correlation coefficient")
    plt.ylim([-30, 30])
    
    if period:
        plt.title('Spearman correlation coefficient of the pageviews ratio with the mobility data ' + str(period) + ' days before and after '+ event.lower().replace('_', ' '))
        plt.savefig("spearman_"+str(period)+"_"+event+".png", bbox_inches = "tight", pad_inches=0.4)
    else:
        plt.title('Spearman correlation coefficient of the pageviews ratio with the mobility data')
        plt.savefig("spearman_total.png", bbox_inches = "tight", pad_inches=0.4)
    
    plt.show()