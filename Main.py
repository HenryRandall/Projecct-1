#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Dependencies and Setup
from census import Census
from us import states
from config import (census_key, gkey)
import us
import gmaps
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
from sodapy import Socrata
import requests
import json
from ipywidgets.embed import embed_minimal_html


# In[5]:


# Make Directories
try:
    os.mkdir('Output')
except:
    pass
try:
    os.mkdir('Output/Data')
except:
    pass
try:
    os.mkdir('Output/Scatter')
except:
    pass
try:
    os.mkdir('Output/Heatmaps')
except:
    pass
try:
    os.mkdir('Output/Barplots')
except:
    pass


# In[6]:


# Call Cencus Data
c = Census(census_key, year=2016)
census_data = c.acs5.get(('B01003_001E', 'B17001_002E','B19013_001E'), {'for': 'county:*'})

# Convert to DataFrame
census_df = pd.DataFrame(census_data)

# Column Reordering
census_df= census_df.rename(columns={'B01003_001E': 'Population',
                                      'B17001_002E': 'Poverty Count',
                                      'B19013_001E': 'Median Household Income',
                                      'state':'State',
                                     'county':'County'})

# Convert Poverty Copunt to Poverty Rate (Poverty Count / Population)
census_df['Poverty Rate'] = 100 *     census_df['Poverty Count'].astype(
        int) / census_df['Population'].astype(int)


# In[7]:


# Clean Census Data - Remove Territories
census_df=census_df[census_df.State != '72']
census_df=census_df.reset_index()

# Call Census data for the population density
census_df['FIPS']=census_df['State']+census_df['County']
url='https://api.census.gov/data/2018/pep/population?get=DENSITY&for=county:*&in=state:*&key='+census_key
response = requests.get(url).json()
column_names = response.pop(0)
density_df=pd.DataFrame(response,columns=column_names)

# Create FIPS reference to merge dataframes on, then merge
density_df['FIPS']=density_df['state']+density_df['county']
merge_df = pd.merge(census_df,density_df, on="FIPS")
merge_df= merge_df.rename(columns={'DENSITY':'Population Density'})
merge_df['Population Density']=merge_df['Population Density'].astype(float)

# Select data for the final census data dataframe
census_df=merge_df[['FIPS','Population','Population Density','Median Household Income','Poverty Rate']]


# In[8]:


# Read in Medicare.gov hospital compare url: https://data.medicare.gov/resource/xubh-q36u.json
dataset='xubh-q36u'
client = Socrata('data.medicare.gov', None)
hospitals = client.get(dataset,limit=6000)
hospitals_df = pd.DataFrame(hospitals)

# Clean Hospital Data - Get rid of unrated hospitals
hospitals_df=hospitals_df[['hospital_name','city','state','county_name','hospital_overall_rating']]
hospitals_df=hospitals_df[hospitals_df.hospital_overall_rating != 'Not Available']
hospitals_df=hospitals_df.reset_index()
hospitals_df=hospitals_df.drop(columns='index')
hospitals_df['hospital_overall_rating'] = hospitals_df['hospital_overall_rating'].astype(float)


# In[27]:


# Define Urls for the Johns Hopkins Data
confirm_url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
death_url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'

# Read in COVID-19 Files
confirm_df=pd.read_csv(confirm_url, error_bad_lines=False)
death_df=pd.read_csv(death_url, error_bad_lines=False)

# Pull State Data
state_confirms=confirm_df.groupby('Province_State').sum()
state_confirms=state_confirms.drop(columns=['UID','code3','FIPS','Lat','Long_'])
state_deaths=death_df.groupby('Province_State').sum()
state_deaths=state_deaths.drop(columns=['UID','code3','FIPS','Lat','Long_'])

state_confirms.to_csv('Output/Data/stateconfirms.csv',index=False,header=True)
state_deaths.to_csv('Output/Data/statedeaths.csv',index=False,header=True)


# In[28]:


# Clean Confirm data Rows - Remove data for territories and cruise ships, format FIPS
confirm_df=confirm_df[confirm_df.Admin2 != 'Unassigned']
confirm_df=confirm_df.dropna()
confirm_df=confirm_df[~confirm_df['Admin2'].astype(str).str.startswith('Out of')]
confirm_df=confirm_df[confirm_df.Admin2 != 'Out of*']
confirm_df=confirm_df.reset_index()
confirm_df['FIPS']=confirm_df.FIPS.map('{0:0>5.0f}'.format)

# Clean Death Data Rows - Remove data for territories and cruise ships
death_df=death_df[death_df.Admin2 != 'Unassigned']
death_df=death_df.dropna()
death_df=death_df[~death_df['Admin2'].astype(str).str.startswith('Out of')]
death_df=death_df[death_df.Admin2 != 'Out of*']
death_df=death_df.reset_index()

# Pull County data
county_confirms=confirm_df.drop(columns=['UID','index','iso2','iso3','code3','FIPS','Lat','Long_'])
county_deaths=death_df.drop(columns=['UID','index','iso2','iso3','code3','FIPS','Lat','Long_'])

county_confirms.to_csv('Output/Data/countyconfirms.csv',index=False,header=True)
county_deaths.to_csv('Output/Data/countydeaths.csv',index=False,header=True)


# In[8]:


# Create the disease DataFrame
disease_df=confirm_df[['FIPS','Admin2','Province_State','Lat','Long_']]
disease_df= disease_df.rename(columns={'Admin2': 'County',
                                      'Province_State': 'State',
                                      'Lat': 'Latitude',
                                      'Long_':'Longitude'})
# Create lists to add case numbers for selected days (5th, 10th, 20th, 50th)
confirm_5=[]
confirm_10=[]
confirm_20=[]
confirm_50=[]
confirm_current=[]
death_5=[]
death_10=[]
death_20=[]
death_50=[]
death_current=[]
[r,c]=confirm_df.shape

# Find case numbers
for j in range (0,r):
    for i in range (12,c):
        if confirm_df.iloc[j, i]>0:
            break
    try:
        response=confirm_df.iloc[j, i+4]
        confirm_5.append(response)
    except:
        confirm_5.append(np.nan)
    try:
        response=confirm_df.iloc[j, i+9]
        confirm_10.append(response)
    except:
        confirm_10.append(np.nan)
    try:
        response=confirm_df.iloc[j, i+19]
        confirm_20.append(response)
    except:
        confirm_20.append(np.nan)
    try:
        response=confirm_df.iloc[j, i+49]
        confirm_50.append(response)
    except:
        confirm_50.append(np.nan)
    try:
        response=death_df.iloc[j, i+5]
        death_5.append(response)
    except:
        death_5.append(np.nan)
    try:
        response=death_df.iloc[j, i+10]
        death_10.append(response)
    except:
        death_10.append(np.nan)
    try:
        response=death_df.iloc[j, i+20]
        death_20.append(response)
    except:
        death_20.append(np.nan)
    try:
        response=death_df.iloc[j, i+50]
        death_50.append(response)
    except:
        death_50.append(np.nan)
    confirm_current.append(confirm_df.iloc[j,c-1])
    death_current.append(death_df.iloc[j,c])

# Add case numbers to the main dataframe
disease_df['confirm_5']=confirm_5
disease_df['confirm_10']=confirm_10
disease_df['confirm_20']=confirm_20
disease_df['confirm_50']=confirm_50
disease_df['confirm_current']=confirm_current
disease_df['death_5']=death_5
disease_df['death_10']=death_10
disease_df['death_20']=death_20
disease_df['death_50']=death_50
disease_df['death_current']=death_current


# In[9]:


#Merge County Census Data with County COVID data
data_df = pd.merge(disease_df, census_df, on="FIPS")

# Format the Hospital datafram to match the census dataframe county and state format
hospitals_df['county_name'] =hospitals_df['county_name'].str.capitalize()
State=[]
for row in hospitals_df.itertuples(index=False):
    State.append(us.states.lookup(row.state).name)
hospitals_df['state']=State

# Save Hospital data to CSV
hospitals_df.to_csv('Output/Data/hospitals.csv',index=False,header=True)

# Average hospital quality over counties and merge with the census dataframe
h_df=hospitals_df[['state','county_name','hospital_overall_rating']]
h_df=h_df.groupby(['county_name','state']).mean().reset_index()
h_df= h_df.rename(columns={'hospital_overall_rating': 'Average Hospital',
                        'county_name':'County',
                        'state':'State'})
county_hospitals_df = data_df.merge(h_df, how='inner', left_on=["State", "County"], right_on=["State","County"])
data_df.to_csv('Output/Data/county_data.csv',index=False,header=True)


# In[10]:


#Create Heat maps
def heatmap(x):
    
    #Configure gmaps with API key
    gmaps.configure(api_key=gkey)
    
    #Removing NAN values
    temp_df=data_df[[x,'Latitude','Longitude']].dropna()
    weights=temp_df[x].astype(float)
    fig = gmaps.figure()
    
    # Create heatmap layer
    heatmap = gmaps.heatmap_layer(temp_df[['Latitude','Longitude']], weights=weights)
    
    # Set heatmap format, use quartiles to set the max intensity for all upper outliers
    heatmap.max_intensity = (float(weights.quantile([.75]))*2.5)-(float(weights.quantile([.25]))*1.5)
    heatmap.dissipating=False
    heatmap.point_radius = .25
    
    # Create and Save Figure
    fig.add_layer(heatmap)
    embed_minimal_html(f'Output/Heatmaps/{x}.html', views=[fig])


# In[11]:


# Create Heatmaps for every variable in the county dataframe
heatmap('confirm_5')
heatmap('confirm_10')
heatmap('confirm_20')
heatmap('confirm_50')
heatmap('death_5')
heatmap('death_10')
heatmap('death_20')
heatmap('death_50')
heatmap('confirm_current')
heatmap('death_current')
heatmap('Population')
heatmap('Population Density')
heatmap('Median Household Income')
heatmap('Poverty Rate')


# In[12]:


#Summary statistics for hospital quality for each state 
state_stats_group=hospitals_df.groupby('state').agg({'hospital_overall_rating': ['mean', 'median', 'var', 'std', 'sem']})

#Average hospital quality for each state
state_stats_group['hospital_overall_rating']['mean'].plot.bar()

#Save file and output to png
plt.savefig('Output/Barplots/state_hospitals.png')

#Create state level dataframe
plt.close()
state_stats_group=pd.DataFrame (state_stats_group)
state_stats_group.to_csv('Output/Data/hospital_state_stats.csv',index=True,header=True)


# In[13]:


def hospital_scatterplot(x,y):
    
    # Pull data and format it correctly for graphing
    plt.figure()
    temp_df=county_hospitals_df[[x,y]]
    temp_df=temp_df.dropna()
    x_values = temp_df[x].astype('float')
    y_values = temp_df[y].astype('float')
    
    # Run linear regression
    (slope, intercept, rvalue, pvalue, stderr) = st.linregress(x_values, y_values)
    regress_values = x_values * slope + intercept
    line_eq = f'y = {str(round(slope,2))} x + {str(round(intercept,2))} \nR squared: {round(rvalue**2,4)}'
    
    # Plot scatter plot
    plt.scatter(x_values,y_values)
    
    # Plot regression line
    plt.plot(x_values,regress_values,"r-")
    plt.annotate(line_eq,(3,np.max(y_values)*.85),fontsize=15,color="red")
    
    # Label plot
    plt.xlabel(x)
    plt.ylabel(y)
    
    #save file and output to png, Make directory if it doesnt already exist
    try:
        os.mkdir(f'Output/Scatter/{x}')
    except:
        pass
    plt.savefig(f'Output/Scatter/{x}/{x}_vs_{y}.png')
    plt.close()


# In[14]:


# Creeate hospital quality scatterplots
hospital_scatterplot('Average Hospital','confirm_5')
hospital_scatterplot('Average Hospital','death_5')
hospital_scatterplot('Average Hospital','confirm_10')
hospital_scatterplot('Average Hospital','death_10')
hospital_scatterplot('Average Hospital','confirm_20')
hospital_scatterplot('Average Hospital','death_20')
hospital_scatterplot('Average Hospital','confirm_50')
hospital_scatterplot('Average Hospital','death_50')


# In[15]:


# Create summary stats for census data of each state
state_stats = data_df.groupby('State').agg({'Population':['mean','median','var','std','sem'],
                                              'Population Density':['mean','median','var','std','sem'],
                                              'Median Household Income':['mean','median','var','std','sem'],
                                              'Poverty Rate':['mean','median','var','std','sem']}
                                         )

# Format and save state summary stats
state_stats = round(state_stats,2)
state_stats.to_csv('Output/Data/state_stats.csv',index=True,header=True)


# In[16]:


# Create summary stats for COVID data of each state
new_df1 = data_df.copy()
clean_df = new_df1.dropna(how='any')
case_stats = new_df1.groupby('State').agg({"confirm_5":['sum','mean','median','var','std','sem'],
                                          "confirm_10":['sum','mean','median','var','std','sem'],
                                          "confirm_20":['sum','mean','median','var','std','sem'],
                                          "confirm_50":['sum','mean','median','var','std','sem'],
                                          "death_5":['sum','mean','median','var','std','sem'],
                                          "death_10":['sum','mean','median','var','std','sem'],
                                          "death_20":['sum','mean','median','var','std','sem'],
                                          "death_50":['sum','mean','median','var','std','sem']
                                         })

# Format and save state summary stats
case_stats = round(case_stats,2)
case_stats.to_csv('Output/Data/case_stats.csv',index=True,header=True)


# In[17]:


def scatterplot(x,y):
    # Pull and format data
    temp_df = data_df[[x,y]]
    temp_df = temp_df.dropna()
    x_values = temp_df[x].astype('float')
    y_values = temp_df[y].astype('float')
    
    # Calculate regression and write annotation
    (slope, intercept, rvalue, pvalue, stderr) = st.linregress(x_values, y_values)
    regress_values = x_values * slope + intercept
    line_eq = f'y = {str(round(slope,2))} x + {str(round(intercept,2))} \nR squared: {round(rvalue**2,4)}'
    
    # Plot scatter plot
    plt.scatter(x_values,y_values)
    
    # Plot regression line
    plt.plot(x_values,regress_values,"r-")
    plt.annotate(line_eq,(np.max(x_values)*.55,np.max(y_values)*.85),fontsize=15,color="red")
    
    # Label plot
    plt.xlabel(x)
    plt.ylabel(y)
    
    #save file and output to png, create directory if needed
    try:
        os.mkdir(f'Output/Scatter/{x}')
    except:
        pass
    plt.savefig(f'Output/Scatter/{x}/{x}_vs_{y}.png')
    plt.close()


# In[18]:


# Call Scatter Plots
scatterplot('Population','confirm_5')
scatterplot('Population','death_5')
scatterplot('Population','confirm_10')
scatterplot('Population','death_10')
scatterplot('Population','confirm_20')
scatterplot('Population','death_20')
scatterplot('Population','confirm_50')
scatterplot('Population','death_50')
scatterplot('Population Density','confirm_5')
scatterplot('Population Density','death_5')
scatterplot('Population Density','confirm_10')
scatterplot('Population Density','death_10')
scatterplot('Population Density','confirm_20')
scatterplot('Population Density','death_20')
scatterplot('Population Density','confirm_50')
scatterplot('Population Density','death_50')
scatterplot('Median Household Income','confirm_5')
scatterplot('Median Household Income','death_5')
scatterplot('Median Household Income','confirm_10')
scatterplot('Median Household Income','death_10')
scatterplot('Median Household Income','confirm_20')
scatterplot('Median Household Income','death_20')
scatterplot('Median Household Income','confirm_50')
scatterplot('Median Household Income','death_50')
scatterplot('Poverty Rate','confirm_5')
scatterplot('Poverty Rate','death_5')
scatterplot('Poverty Rate','confirm_10')
scatterplot('Poverty Rate','death_10')
scatterplot('Poverty Rate','confirm_20')
scatterplot('Poverty Rate','death_20')
scatterplot('Poverty Rate','confirm_50')
scatterplot('Poverty Rate','death_50')


# In[19]:


def barplot(x,y):
    
    # Pull data and sum
    state_data = data_df.groupby(x)
    case_count = state_data[y].sum()
    
    # Barplot
    data_count_df = pd.DataFrame({'Number of Cases': case_count})
    data_count_df.sort_values(by='Number of Cases', ascending=False).plot(kind="bar", align="center", legend=False, width=0.75, figsize=(12,8))
    plt.xlim(-0.75, len(data_count_df)-0.25)
    plt.ylim(0, max(data_count_df["Number of Cases"])*1.03)
    
    # Set a Title and labels
    plt.title("Number of Cases per State")
    plt.ylabel(y)
    
    # Save plot as png, make a directory if neede
    try:
        os.mkdir(f'Output/Barplots/{x}')
    except:
        pass
    plt.savefig(f'Output/Barplots/{x}/{x}_vs_{y}.png')
    plt.close()


# In[20]:


# Call barplot
barplot('State','confirm_5')
barplot('State','death_5')
barplot('State','confirm_10')
barplot('State','death_10')
barplot('State','confirm_20')
barplot('State','death_20')
barplot('State','confirm_50')
barplot('State','death_50')


# In[ ]:




