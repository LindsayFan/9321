#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math
import re

studentid = os.path.basename(sys.modules[__name__].__file__)


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


def question_1(exposure, countries):
    """
    :param exposure: the path for the exposure.csv file
    :param countries: the path for the Countries.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    # exposure.csv seperate by ;   
    df_exposure = pd.read_csv(exposure, sep = ';')
    # delete all rows without country
    df_exposure = df_exposure[df_exposure['country'].notnull()]
    # change the column 'country' to 'Country' in order to match to 'countries.csv'
    df_exposure.rename(columns = {'country':'Country'},inplace = True)

    # countries.csv: seperate each row s by ','
    df_countries = pd.read_csv(countries, sep = ',')

    # I diaplace all the unique countries from countries.csv and exposure.csv and change them manually.
    df_exposure.replace('Brunei Darussalam', 'Brunei',inplace = True)
    df_exposure.replace('Cabo Verde', 'Cape Verde',inplace = True)
    df_countries.replace('Republic of the Congo', 'Congo',inplace = True)
    df_countries.replace('Democratic Republic of the Congo','Congo DR' ,inplace = True)
    df_exposure.replace( "C魌e d'Ivoire", "Ivory Coast",inplace = True)
    df_exposure.replace('Eswatini', 'Swaziland',inplace = True)
    df_exposure.replace('Korea DPR', 'North Korea',inplace = True)
    df_exposure.replace('Korea Republic of', 'South Korea',inplace = True)
    df_exposure.replace('Lao PDR', 'Laos',inplace = True)
    df_exposure.replace('Moldova Republic of', 'Moldova',inplace = True)                 
    df_exposure.replace('North Macedonia', 'Macedonia',inplace = True)
    df_exposure.replace('Palestine', 'Palestinian Territory',inplace = True)
    df_countries.replace('Russia', 'Russian Federation',inplace = True)
    df_exposure.replace( 'United States of America', 'United States',inplace = True)
    df_exposure.replace('Viet Nam', 'Vietnam',inplace = True)

    # merged by column "Country"
    df1 = pd.merge(df_exposure,df_countries, on = "Country")

    # Change countries' name and let them as the same as the value in "countries-contonents.csv"
    df1.replace('Burkina Faso', 'Burkina',inplace = True)
    df1.replace('Myanmar', "Burma (Myanmar)",inplace = True)
    df1.replace('Czech Republic', 'CZ',inplace = True)
    df1.replace('United States', 'US',inplace = True)
    df1.replace('Congo DR', 'Congo, Democratic Republic of',inplace = True)
    df1.replace('North Korea', 'Korea, North',inplace = True)
    df1.replace('South Korea', 'Korea, South',inplace = True)

    # set column 'Country' as index
    df1 = df1.set_index('Country')

    # sort by Country 
    df1.sort_values(by=['Country'],ascending = True,inplace = True)

    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    df2 = df1.copy()
    avg_latitude = []
    avg_longitude = []
    for i in range(0,df2.shape[0]): # loop each rows
        df2_cities = df2.iloc[i].at['Cities']
        a = df2_cities.split("|||")
        latitude = []
        longitude = []
        for i in a:     # loop each string that is splites by "|||" in column"Cities
            dic = json.loads(i)  # change each value from string to a dictionary
            latitude.append(dic["Latitude"])
            longitude.append(dic["Longitude"])
        avg_latitude.append(np.mean(latitude))
        avg_longitude.append(np.mean(longitude))
    # assign the ave value to the dataframe
    df2["avg_latitude"]= avg_latitude
    df2["avg_longitude"]= avg_longitude

    log("QUESTION 2", output_df=df2[["avg_latitude", "avg_longitude"]], other=df2.shape)
    return df2

def geo_distance(lon2, lat2):  
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1= 30.5928  # The coordinates of Wuhan
    lon1 = 114.3055
    r = 6373   # Earth's radios
    lon1, lat1, lon2, lat2 = map(math.radians, map(float, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return c * r

def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    df3 = df2.copy()
    # call the function geo_distance
    df3['distance_to_Wuhan'] = df3.apply(lambda x: geo_distance(x.avg_longitude, x.avg_latitude), axis=1)

    log("QUESTION 3", output_df=df3[['distance_to_Wuhan']], other=df3.shape)
    return df3


def question_4(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    df_continents = pd.read_csv(continents, sep = ',')
    

    df = df2.copy()
    exposure=[]
    
    # replace "2,1" to "2.1"
    for i in range(0,df.shape[0]):
        df_exposure = df.iloc[i].at['Covid_19_Economic_exposure_index']
        df_exposure = df_exposure.replace(",", ".")
        exposure.append(df_exposure)

    df.drop('Covid_19_Economic_exposure_index',inplace = True, axis = 1)   
    df['Covid_19_Economic_exposure_index'] = exposure
    df['Covid_19_Economic_exposure_index'] = pd.to_numeric(df['Covid_19_Economic_exposure_index'],errors='coerce')

    #Merge two dataset
    df_merge = pd.merge(df_continents,df, on = "Country")
    
    # The result dataframe
    data = {'Continent': df_merge["Continent"], 
            'Country': df_merge["Country"], 
            'Covid_19_Economic_exposure_index': df_merge["Covid_19_Economic_exposure_index"], 
    }
    df_new = pd.DataFrame(data, columns = ['Continent', 'Country', 'Covid_19_Economic_exposure_index'])
    
    df4 = df_new.groupby("Continent").mean()
    
    df4.rename(columns = {"Covid_19_Economic_exposure_index":"average_covid_19_Economic_exposure_index"}, inplace=True)
    df4.sort_values(by=['average_covid_19_Economic_exposure_index'],ascending = True,inplace = True)

    log("QUESTION 4", output_df=df4, other=df4.shape)
    return df4


def question_5(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df5
            Data Type: dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    df_5 = df2.copy()
    # farmat the column['Foreign direct investment']
    foreign=[]
    for i in range(0,df_5.shape[0]):
        df_foreign = df_5.iloc[i].at['Foreign direct investment']
        df_foreign = df_foreign.replace(",", ".")
        foreign.append(df_foreign)

    df_5.drop('Foreign direct investment',inplace = True, axis = 1)   
    df_5['Foreign direct investment'] = foreign
    df_5['Foreign direct investment'] = pd.to_numeric(df_5['Foreign direct investment'],errors='coerce')

    # farmat the columns['Net_ODA_received_perc_of_GNI']
    gni=[]
    for i in range(0,df_5.shape[0]):
        df_gni = df_5.iloc[i].at['Net_ODA_received_perc_of_GNI']
        df_gni = df_gni.replace(",", ".")
        gni.append(df_gni)
    df_5.drop('Net_ODA_received_perc_of_GNI',inplace = True, axis = 1)   
    df_5['Net_ODA_received_perc_of_GNI'] = gni
    df_5['Net_ODA_received_perc_of_GNI'] = pd.to_numeric(df_5['Net_ODA_received_perc_of_GNI'],errors='coerce')

    
    data5 = {'Income Class': df_5["Income classification according to WB"], 
            'Avg Foreign direct investment': df_5["Foreign direct investment"], 
            'Avg_ Net_ODA_received_perc_of_GNI': df_5["Net_ODA_received_perc_of_GNI"], 
    }
    df_new5 = pd.DataFrame(data5, columns = ['Income Class', 'Avg Foreign direct investment', 'Avg_ Net_ODA_received_perc_of_GNI'])
    df_new5 = df_new5.reset_index(drop=True)
    df5 = df_new5.groupby("Income Class").mean()

    log("QUESTION 5", output_df=df5, other=df5.shape)
    return df5


def question_6(df2):
    """
    :param df2: the dataframe created in question 2
    :return: cities_lst
            Data Type: list
            Please read the assignment specs to know how to create the output dataframe
    """
    cities_lst = []
    
    df_6 = df2.copy()
    
    df_all_po = pd.DataFrame()
    for i in range(0,df_6.shape[0]):
        df_population = df_6.iloc[i].at['Cities']
        a6 = df_population.split("|||")
        population=[]
        city = []
        country = []
        for i in a6:     
            dic = json.loads(i)
            population.append(dic["Population"])
            city.append(dic["City"])
            country.append(dic["Country"])
        series_po = pd.Series(population)
        series_ci = pd.Series(city)
        series_co = pd.Series(country)
        frame_new6 = {'Country':series_co,'city':series_ci,'population':series_po}
        df_new6 = pd.DataFrame(frame_new6)
        df_new6 = df_new6[df_new6['population'].notnull()]
        df_new6 = df_new6.reset_index(drop = True)
        df_all_po = pd.concat([df_all_po,df_new6],ignore_index = True)

    incomeclass = {'income class':df_6['Income classification according to WB']}
    df_incomeclass = pd.DataFrame(incomeclass,columns = ['income class'])
    
    df_all6 = pd.merge(df_all_po,df_incomeclass, on = "Country")
    df_all6 = df_all6[df_all6['income class'] == "LIC"]
    df_all6.sort_values(by=['population'],ascending = False, inplace = True)
    df_all6 = df_all6.reset_index(drop =True)

    for i in range(0,5):
        lst = df_all6.iloc[i].at['city']
        cities_lst.append(lst)

    log("QUESTION 6", output_df=None, other=cities_lst)
    return cities_lst


def question_7(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    #################################################

    log("QUESTION 7", output_df=df7, other=df7.shape)
    pass 


def question_8(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: nothing, but saves the figure on the disk
    """

    df_continents = pd.read_csv(continents, sep = ',')
    df_8 = df2.copy()
    total_population = []
    for i in range(0,df_8.shape[0]):
        df_population = df_8.iloc[i].at['Cities']
        a = df_population.split("|||")
        population=[]
        for i in a:     
            dic = json.loads(i)
            if not dic["Population"] is None:
                population.append(dic["Population"])
        total_population.append(sum(population))
    df_8["population"] = total_population


    df_new8 =pd.merge(df_continents,df_8, on = "Country")


    # find the the total population s of the world
    lis=[]
    for i in range(0,df_new8.shape[0]):
        population = df_new8.iloc[i].at['population']
        lis.append(population)
    s = sum(lis)

    # select all the row
    df8=df_new8.loc[df_new8['Continent']=='South America'] 
    df8 = df8.reset_index(drop=True)


    percentage_lis=[]
    for i in range(0,df8.shape[0]):
        df_percentage = df8.iloc[i].at['population']
        percentage = df_percentage/s
        percentage_lis.append(percentage)
    df8['The percentage of population'] = percentage_lis
    df8 = df8.iloc[:,[1,27]]

    df8.plot(x='Country',y = 'The percentage of population',kind='bar',
             yticks=([0,0.0025,0.005,0.0075,0.01,0.015,0.02,0.07]),legend=False,figsize=(10,8))
    plt.title('The percentage of world population in each South American country',fontsize= 18)
    plt.grid(color="k", linestyle=":")
    plt.xlabel("Countries in South America")
    plt.ylabel("The percentage of the world population")
    

    plt.savefig("{}-Q11.png".format(studentid))


def question_9(df2):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    """

    df_9 = df2.copy()

    fdi=[]
    for i in range(0,df_9.shape[0]):
        df_fdi = df_9.iloc[i].at['Covid_19_Economic_exposure_index_Ex_aid_and_FDI']
        df_fdi = df_fdi.replace(",", ".")
        fdi.append(df_fdi)
    df_9.drop('Covid_19_Economic_exposure_index_Ex_aid_and_FDI',inplace = True, axis = 1)   
    df_9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'] = fdi
    df_9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'] = pd.to_numeric(df_9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'],
                                                                            errors='coerce')

    foodimport=[]
    for i in range(0,df_9.shape[0]):
        df_foodimport = df_9.iloc[i].at['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import']
        df_foodimport = df_foodimport.replace(",", ".")
        foodimport.append(df_foodimport)
    df_9.drop('Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import',inplace = True, axis = 1)   
    df_9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'] = foodimport
    df_9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'] = pd.to_numeric(
        df_9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'],errors='coerce')

    gdp=[]
    for i in range(0,df_9.shape[0]):
        df_gdp = df_9.iloc[i].at['Foreign direct investment, net inflows percent of GDP']
        df_gdp = df_gdp.replace(",", ".")
        gdp.append(df_gdp)
    df_9.drop('Foreign direct investment, net inflows percent of GDP',inplace = True, axis = 1)   
    df_9['Foreign direct investment, net inflows percent of GDP'] = gdp
    df_9['Foreign direct investment, net inflows percent of GDP'] = pd.to_numeric(
        df_9['Foreign direct investment, net inflows percent of GDP'],errors='coerce')

    foreign=[]
    for i in range(0,df_9.shape[0]):
        df_foreign = df_9.iloc[i].at['Foreign direct investment']
        df_foreign = df_foreign.replace(",", ".")
        foreign.append(df_foreign)
    df_9.drop('Foreign direct investment',inplace = True, axis = 1)   
    df_9['Foreign direct investment'] = foreign
    df_9['Foreign direct investment'] = pd.to_numeric(df_9['Foreign direct investment'],errors='coerce')

    # The result dataframe
    data9 = {'Income Class': df_9["Income classification according to WB"],
             'Foreign direct investment, net inflows percent of GDP':df_9['Foreign direct investment, net inflows percent of GDP'],
            'Foreign direct investment': df_9["Foreign direct investment"],
             'Covid_19_Economic_exposure_index_Ex_aid_and_FDI':df_9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'],
             'Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import':df_9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import']
             }
    df_new9 = pd.DataFrame(data9, columns = ['Income Class', 'Covid_19_Economic_exposure_index_Ex_aid_and_FDI',
                                             'Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import',
                                             'Foreign direct investment, net inflows percent of GDP',
                                            'Foreign direct investment'])
    # fill all the none with zero
    df_new9.fillna(0,inplace = True) 
    df9 = df_new9.groupby('Income Class').sum()
    
    # I want the X-axis to be in the order of HIC, MIC , LIC
    index = ['3','1','2']
    df9['index'] = index
    df9.sort_values("index",ascending = False,inplace = True)
    
    # plot stacked bar graph
    df9.plot(kind='bar',figsize=(17,9),grid = True,stacked = True)
    plt.legend(loc = 'upper right',fontsize=10)
    plt.title('The differenies among the high, middle and low income countries',fontsize=18)
    plt.xlabel("Income level",fontsize=14)
    plt.ylabel("Quantity of each index",fontsize=14)

    plt.savefig("{}-Q12.png".format(studentid))


def question_10(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    :param continents: the path for the Countries-Continents.csv file
    """

    df_continents = pd.read_csv(continents, sep = ',')
    df_10 = df2.copy()
    total_population = []
    for i in range(0,df_10.shape[0]):
        df_population = df_10.iloc[i].at['Cities']
        a = df_population.split("|||")
        population=[]
        for i in a:     
            dic = json.loads(i)
            if not dic["Population"] is None:
                population.append(dic["Population"])
        total_population.append(sum(population))
    df_10["population"] = total_population


    df_new10 =pd.merge(df_continents,df_8, on = "Country")
    df10 = df_new10.iloc[:,[0,1,24,25,26]]
    df10 = df10.set_index('Continent')
    
    # change the big dataframe in to small one
    df10_africa = df10.loc["Africa"]
    df10_south = df10.loc['South America']
    df10_asia = df10.loc['Asia']
    df10_europe = df10.loc['Europe']
    df10_oceania = df10.loc['Oceania']
    df10_north = df10.loc['North America']


    Africa = df10_africa.plot.scatter(x='avg_longitude', y='avg_latitude', s=df10_africa['population']*0.00003, 
                                      color='orange', label = 'Africa',figsize=(20,10))
    South_America = df10_south.plot.scatter(x='avg_longitude', y='avg_latitude', s=df10_south['population']*0.00003,
                                            label = 'South America',color='Green',ax = Africa)
    Asia = df10_asia.plot.scatter(x='avg_longitude', y='avg_latitude',s=df10_asia['population']*0.00003,
                                  label = 'Asia', color = 'Red',ax = South_America)
    Europe= df10_europe.plot.scatter(x='avg_longitude', y='avg_latitude',s=df10_europe['population']*0.00003,
                                     label = 'Europe', color = 'Blue',ax = Asia)
    Oceania= df10_oceania.plot.scatter(x='avg_longitude', y='avg_latitude',s=df10_oceania['population']*0.00003, 
                                       label = 'Oceania' , color = 'purple',ax = Europe)
    North_America= df10_north.plot.scatter(x='avg_longitude', y='avg_latitude',s=df10_north['population']*0.00003,
                                           label = 'North America',color = 'cyan',ax = Oceania)
    plt.legend(loc = 'upper left',fontsize=20)
    plt.title('The distribution of population in the country of each continents',fontsize=18)
    plt.xlabel("Longitude",fontsize=14)
    plt.ylabel("Laititude",fontsize=14)

    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("exposure.csv", "Countries.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df2.copy(True))
    df4 = question_4(df2.copy(True), "Countries-Continents.csv")
    df5 = question_5(df2.copy(True))
    lst = question_6(df2.copy(True))
    df7 = question_7(df2.copy(True))
    question_8(df2.copy(True), "Countries-Continents.csv")
    question_9(df2.copy(True))
    question_10(df2.copy(True), "Countries-Continents.csv")

