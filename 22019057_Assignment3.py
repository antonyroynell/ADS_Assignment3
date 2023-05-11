import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
from scipy.optimize import curve_fit


def exponential_func(t, n0, g):
    t = t - 1990
    f = n0*np.exp(g*t)
    return f


def err_ranges(year, exponential_func, popt, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = exponential_func(year, *popt)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(popt, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = exponential_func(year, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


def greenhouse_emissions_forecast_func(df):
    df = df[['Country Name', '1990 [YR1990]', '1995 [YR1995]', '2000 [YR2000]',
             '2005 [YR2005]', '2010 [YR2010]',  '2015 [YR2015]', '2019 [YR2019]']]
    df.columns = ['Country Name', '1990', '1995',
                  '2000', '2005', '2010', '2015',  '2020']
    li = df.columns.values
    for i in li:
        df[i].replace('..', np.nan, inplace=True)
    df = df.dropna()
    li = df.columns.values
    df = df[df['Country Name'] == 'China']
    df_t = pd.DataFrame.transpose(df)
    df_t = df_t.tail(-1)
    df_t.reset_index(inplace=True)
    df_t.columns = ['Year', 'GH Emission']
    df_t['Year'] = df_t['Year'].astype(int)
    df_t['GH Emission'] = df_t['GH Emission'].astype(float)
    plt.plot(df_t['Year'], df_t['GH Emission'],
             label='Actual Greenhouse Emissions')
    plt.xlabel('Years', **font2, c = 'black')
    plt.ylabel('GreenHouse Emissions (kt)', **font2, c = 'black')
    plt.title('Greenhouse Emissions of China & forecast', **font1, c = 'black')
    popt, pcov = curve_fit(
        exponential_func, df_t['Year'], df_t['GH Emission'], p0=(0.27e7, 0.056))
    print("GH EMISSIONS 1990", popt[0]/1e9)
    #plt.plot(df_t['Year'], exponential_func(
        #df_t['Year'], *popt), label='Curved Fit')
    year = np.linspace(1990, 2030, 5)
    forecast = exponential_func(year, *popt)
    plt.plot(year, forecast, label='Forecast curve',
             linewidth=4, alpha=0.6, color='green')
    sigma = np.sqrt(np.diag(pcov))
    low, up = err_ranges(year, exponential_func, popt, sigma)
    plt.fill_between(year, low, up, color='yellow', alpha=0.7, label='Confidence Range')
    plt.legend(loc = 'upper left', **font4)
    plt.xticks(**font3,  c ='black')
    plt.yticks(**font3, c ='black')
    plt.savefig('greenhouse_emissions.png', dpi = 500)
    plt.show()
    

def scaler_func(df):
    scaler = MinMaxScaler()
    scaler.fit(df[['1990']])
    df['1990'] = scaler.transform(df[['1990']])
    scaler.fit(df[['2020']])
    df['2020'] = scaler.transform(df[['2020']])
    return df


def silhouette_score_func(df):
    score = []
    n_of_cluster = []
    for n in range(2, 12):
        km = KMeans(n_clusters=n)
        km.fit(df)
        labels = km.labels_
        cen = km.cluster_centers_
        score.append(skmet.silhouette_score(df, labels))
        n_of_cluster.append(n)
    return n_of_cluster[score.index(max(score))], max(score)


def co2_emissions_cluster_func(df):
    df = df[['1990 [YR1990]', '2019 [YR2019]']]
    df.columns = ['1990', '2020']
    li = df.columns.values
    for i in li:
        df[i].replace('..', np.nan, inplace=True)
    df = df.dropna()
    li = df.columns.values
    for i in li:
        df[i] = df[i].astype(float)
    df = scaler_func(df)
    n_cluster, silhouette_score = silhouette_score_func(df)
    print('\t Clustering CO2 EMissions in (kt)')
    print('Maximum Silhouette Score is : ', +silhouette_score)
    print('Number of Clusters is : ', +n_cluster)
    km = KMeans(n_clusters=n_cluster)
    y_predicted = km.fit_predict(df[['1990', '2020']])
    df['cluster'] = y_predicted
    df_cluster1 = df[df.cluster == 0]
    df_cluster2 = df[df.cluster == 1]
    df_cluster3 = df[df.cluster == 2]
    plt.scatter(df_cluster1['1990'], df_cluster1['2020'],
                color='green', label='Cluster1')
    plt.scatter(df_cluster2['1990'], df_cluster2['2020'],
                color='red', label='Cluster2')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
                :, 1], color='purple', marker='*', label='Centroid')
    plt.xlabel('1990', **font2, c = 'black')
    plt.ylabel('2020', **font2, c= 'black')
    plt.title('CO2 Emissions (kt)', **font1, c = 'black')
    plt.xticks(**font3,  c ='black')
    plt.yticks(**font3, c ='black')
    plt.legend(**font4)
    plt.savefig('c02_emissions.png', dpi = 500)
    plt.show()


def methane_emissions_cluster_func(df):
    df = df[['1990 [YR1990]', '2019 [YR2019]']]
    df.columns = ['1990', '2020']
    li = df.columns.values
    for i in li:
        df[i].replace('..', np.nan, inplace=True)
    df = df.dropna()
    li = df.columns.values
    for i in li:
        df[i] = df[i].astype(float)
    df = scaler_func(df)
    n_cluster, silhouette_score = silhouette_score_func(df)
    print('\t Clustering Methane EMissions in (kt)')
    print('Maximum Silhouette Score is : ', +silhouette_score)
    print('Number of Clusters is : ', +n_cluster)
    km = KMeans(n_clusters=n_cluster)
    y_predicted = km.fit_predict(df[['1990', '2020']])
    df['cluster'] = y_predicted
    df_cluster1 = df[df.cluster == 0]
    df_cluster2 = df[df.cluster == 1]
    df_cluster3 = df[df.cluster == 2]
    plt.scatter(df_cluster1['1990'], df_cluster1['2020'],
                color='green', label='Cluster1')
    plt.scatter(df_cluster2['1990'], df_cluster2['2020'],
                color='red', label='Cluster2')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
                :, 1], color='purple', marker='*', label='Centroid')
    plt.xlabel('1990', **font2, c = 'black')
    plt.ylabel('2020', **font2, c = 'black')
    plt.title('Methane Emissions (kt)', **font1, c = 'black')
    plt.xticks(**font3,  c ='black')
    plt.yticks(**font3, c ='black')
    plt.legend(**font4)
    plt.savefig('methane_emissions.png', dpi = 500)
    plt.show()


def agri_forest_land_cluster_func(df1, df2):
    df1 = df1[['2019 [YR2019]']]
    df1.columns = ['Agri_Land']
    df1['Agri_Land'].replace('..', np.nan, inplace=True)
    df1 = df1.dropna()
    df1['Agri_Land'] = df1['Agri_Land'].astype(float)
    df1 = df1.head(25)
    df1.reset_index(inplace=False)
    agri_land = df1['Agri_Land'].tolist()
    df2 = df2[['2019 [YR2019]']]
    df2.columns = ['Forest_Land']
    df2['Forest_Land'].replace('..', np.nan, inplace=True)
    df2 = df2.dropna()
    df2['Forest_Land'] = df2['Forest_Land'].astype(float)
    df2 = df2.head(25)
    df2.reset_index(inplace=False)
    forest_land = df2['Forest_Land'].tolist()
    df = pd.DataFrame()
    df['Forest_Land'] = forest_land
    df['Agri_Land'] = agri_land
    scaler = MinMaxScaler()
    scaler.fit(df[['Forest_Land']])
    df['Forest_Land'] = scaler.transform(df[['Forest_Land']])
    scaler.fit(df[['Agri_Land']])
    df['Agri_Land'] = scaler.transform(df[['Agri_Land']])
    n_cluster, silhouette_score = silhouette_score_func(df)
    print('\t Clustering Agriculture Land vs Forest Land')
    print('Maximum Silhouette Score is : ', +silhouette_score)
    print('Number of Clusters is : ', +n_cluster)
    km = KMeans(n_clusters=n_cluster)
    y_predicted = km.fit_predict(df[['Forest_Land', 'Agri_Land']])
    df['cluster'] = y_predicted
    df_cluster1 = df[df.cluster == 0]
    df_cluster2 = df[df.cluster == 1]
    df_cluster3 = df[df.cluster == 2]
    df_cluster4 = df[df.cluster == 3]
    plt.scatter(df_cluster1['Forest_Land'],
                df_cluster1['Agri_Land'], color='green', label='Cluster1')
    plt.scatter(df_cluster2['Forest_Land'],
                df_cluster2['Agri_Land'], color='red', label='Cluster2')
    plt.scatter(df_cluster3['Forest_Land'],
                df_cluster3['Agri_Land'], color='blue', label='Cluster3')
    plt.scatter(df_cluster4['Forest_Land'],
                df_cluster4['Agri_Land'], color='yellow', label='Cluster4')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
                :, 1], color='purple', marker='*', label='Centroid')
    plt.xlabel('Forest Land', **font2,  c ='black')
    plt.ylabel('Agriculture Land', **font2,  c ='black')
    plt.title('Forest land vs Agri. land', **font1, c ='black')
    plt.xticks(**font3,  c ='black')
    plt.yticks(**font3, c ='black')
    plt.legend(bbox_to_anchor=(1.001, 1),**font4)
    plt.savefig('agri_vs_forest.png', dpi = 500)
    plt.show()


def read_files():
    greenhouse_emissions = pd.read_csv('Total_GreenHouse.csv')
    co2_emissions = pd.read_csv('CO2_Emissions.csv')
    methane_emissions = pd.read_csv('Methane_Emissions.csv')
    forest_area = pd.read_csv('Forest_Area.csv')
    agri_area = pd.read_csv('Agricultural_Area.csv')
    greenhouse_emissions_forecast_func(greenhouse_emissions)
    co2_emissions_cluster_func(co2_emissions)
    methane_emissions_cluster_func(methane_emissions)
    agri_forest_land_cluster_func(agri_area, forest_area)
    
font1 = {'fontsize' : 16}
font2 = {'fontsize' : 14}    
font3 = {'fontsize' : 13}
font4 = {'fontsize' : 11}

plt.style.use('ggplot') 
read_files()
