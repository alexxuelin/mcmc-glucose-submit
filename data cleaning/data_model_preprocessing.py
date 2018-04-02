
# coding: utf-8

# Author: Steven Menelly
# 
# Purpose: Convert Raw BSG data from Dexcom Clarity Service into a workable dataframe
# 
# Deliverable: EDA Graphs and tables

# In[1]:


get_ipython().magic(u'matplotlib inline')

import numpy as np
import pandas as pd
import datetime

import matplotlib
import matplotlib.pyplot as plt



# In[2]:


df = pd.read_csv("bsg_final.csv")
dfalt = pd.read_csv('with_indicator.csv')


# In[3]:


print(df.columns)
print(dfalt.columns)


# In[4]:


bsg = "Glucose Value (mg/dL)"
days_of_week = df["Day of the week"].unique()
print(days_of_week)


# In[5]:


#The dict of the all the days separately

by_day = { weekday : df.loc[lambda df: df["Day of the week"] == weekday,:].copy() for weekday in days_of_week}


# In[6]:



index =[]

weekday_stats = pd.DataFrame([ {"Day of the week":weekday, "Mean score for that day": by_day[weekday][bsg].mean(), "Standard Deviation": by_day[weekday][bsg].std(),"Count": by_day[weekday][bsg].count() } for weekday in days_of_week ])

weekday_stats = pd.concat([weekday_stats,weekday_stats.iloc[[0]]])

weekday_stats = weekday_stats[1:].reset_index()

weekday_stats=weekday_stats.drop('index',axis =1)

cols = weekday_stats.columns.tolist()

cols = cols[1:]+[cols[0]]

weekday_stats = weekday_stats[cols]

print(weekday_stats)



# In[7]:



#Plot of total set

#fig, ax = plt.subplots(1,2 ,figsize = (10,6))

#make two plots, one with the cont values

#plt.title("Weekday Descriptive Statistics" )

#ax[0].set_title('Average glucose levels throughout weekdays')
#weekday_stats.plot(x='Day of the week',y='Mean score for that day', yerr = 'Standard Deviation',kind='bar', ax=ax[0]) 
#ax[0].set_ylabel('Blood Glucose (mg/dL)')
#plt.set_xticks(rotation=45)

#ax[1].set_title("Composition of Data")

#plt.pie(weekday_stats['Count'],labels = weekday_stats['Day of the week'])

#plt.sca(ax[0])
#plt.xticks(rotation=45)


# In[8]:


#creating day column
def label_day(row):
    return row['Timestamp (YYYY-MM-DDThh:mm:ss)'][:10]
def label_time(row):
    return row['Timestamp (YYYY-MM-DDThh:mm:ss)'][11:]
def abs_time(row):
    return (int(row['Time'][:2])*3600)+(int(row['Time'][3:5])*60)+(int(row['Time'][6:]))

def lunch(row):
    if row['Offset']>=(144) and row['Offset'] <=(204):
        return 1
    else:
        return 0


def dinner(row):
    if row['Offset']>=(204) and row['Offset'] <=(264):
        return 1
    else:
        return 0



    
#Without offset
#df['Date'] = df.apply( lambda row: label_day (row), axis =1 )
#df['Time'] = df.apply( lambda row: label_time (row), axis =1 )

#df['Abs Time'] = df.apply( lambda row: abs_time (row), axis =1)

#df['LunchPeriod'] = df.apply( lambda row: lunch (row), axis =1)
#df['DinnerPeriod'] = df.apply( lambda row: dinner (row), axis =1)

dfalt['Date'] = dfalt.apply( lambda row: label_day (row), axis =1 )
dfalt['Time'] = dfalt.apply( lambda row: label_time (row), axis =1 )

dfalt['Abs Time'] = dfalt.apply( lambda row: abs_time (row), axis =1)

dfalt['LunchPeriod'] = dfalt.apply( lambda row: lunch (row), axis =1)
dfalt['DinnerPeriod'] = dfalt.apply( lambda row: dinner (row), axis =1)




#df['Abs Time'] = 

#df['Day'] = df,apply(['Timestamp (YYYY-MM-DDThh:mm:ss)']


# In[9]:


#dates_in_sample = df["Date"].unique()
#by_date = { date : df.loc[lambda df: df["Date"] == date,:].copy() for date in dates_in_sample}

dates_in_sample = dfalt["Date"].unique()
by_date = { date : dfalt.loc[lambda dfalt: dfalt["Date"] == date,:].copy() for date in dates_in_sample}




# In[10]:




#fig, ax = plt.subplots(figsize = (15,10))

#color_dict = {'Monday': 'b' ,'Tuesday': 'g' , 'Wednesday': 'r' , 'Thursday': 'c'}

#dates = list(by_date.keys())

#plt.title('Raw Blood Glucose Data : M-Th, {} through {}, n = {}'.format(dates[0],dates[-1],len(dates)))

#times = ['09:00:00','12:00:00','17:00:00']

#for date, series in by_date.items():
#        day =series['Day of the week'].unique()[0]
#        plt.plot(series['Abs Time'],series['Glucose Value (mg/dL)'] , color = color_dict[day], alpha =0.5)
#plt.tick_params(
#    axis='x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom='off',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#    labelbottom='off') # labels along the bottom edge are off

#plt.xlabel('Time of day- 00:00:00 through 24:00:00')
#plt.ylabel('Blodd Glucose Concentration (mg/dL)')
#for time in times:
#    x = int(time[:2])*3600
#    plt.axvline(x = x, color = 'k', linestyle ='--')
#    plt.text(x, 375, ' t= '+time)
    


# In[11]:


clipped_by_time_lunch = { date: series.loc[lambda series: series['LunchPeriod']==1] for date, series in by_date.items()}

clipped_by_time_dinner = { date: series.loc[lambda series: series['DinnerPeriod']==1] for date, series in by_date.items()}




# In[12]:


import csv
lunchtimes = pd.concat(clipped_by_time_lunch.values())

dinnnertimes  =pd.concat(clipped_by_time_dinner.values())

results = lunchtimes.to_csv('lunches.csv')

results2 = dinnnertimes.to_csv('dinners.csv')





# In[13]:


#Determination of excess glucose

def make_excess(row):
    #return np.maximum([row[bsg]-100],[100])
    return np.maximum(row[bsg]-100,0)
df['Excess Glucose'] = df.apply(lambda row: make_excess(row), axis =1)

df


# Now the actual modeling of meal times
# 

# In[14]:


#Bodily Parameters
df_m = pd.DataFrame()
df_m['Abs Time'] = range(0, 3600*24)


df_m['Glucose from food (mg)']=0 #Assume a fasting state
df_m['M1 Excess Glucose']=0 #Assume a steady state of zero

total_blood_volume = 66.36 # in deciliters
gram_glucose_per_carb = .2765 # Estimated from prescribed information for patient.
                         # Insulin to carb ratio is 1:6 grams, Correction factor
                         # is 1:25 1 unit of insulin is 25 mg/dL.
#digestion_period = 7200 # this is two hours, after which absorption stops
digestion_period = 7200 # this is two hours, after which absorption stops

constant = .0004 #decay constant of glucose in the system from insulin activity

meal_dict = {'Lunch':('12:00:00',30), 'Dinner':('17:00:00',60)} #the time and carb counts for each meal

meal_dict = {key: (value[0],value[1]*gram_glucose_per_carb) for key, value in meal_dict.items()} # converting grams of carbs consumed
                                                                                                 # into grams of glucose consumed


# In[15]:


##Excess Glucose Modeling

def meal_bolus(meal):
    start_time  = meal_dict[meal][0]
    start_time = (int(start_time[:2])*3600)+(int(start_time[3:5])*60)+(int(start_time[6:]))
    end_time = start_time + digestion_period
    delta_glucose = (meal_dict[meal][1]*1000)/digestion_period # convert to miligrams of glucose per second
    def apply_meal(row):
        if start_time<row['Abs Time']<end_time:
            return delta_glucose
        elif row['Glucose from food (mg)']>0:
            return row['Glucose from food (mg)']
        else:
            return
    df_m["Glucose from food (mg)"]= df_m.apply( lambda row: apply_meal(row), axis =1)
    return



meal_bolus('Lunch')
meal_bolus('Dinner')
    
df_m["Glucose from food (mg)"]=df_m["Glucose from food (mg)"].fillna(0)
    


 


# In[16]:



def model_glucose(row):
    index = row['Abs Time']
    if index != 0: #not the first time step - skip
        delta = df_m['Glucose from food (mg)'].iloc[int(index)] - (constant * df_m['M1 Excess Glucose'].iloc[int(index-1)])
        return df_m['M1 Excess Glucose'].iloc[int(index-1)] + len(df)*(delta)
        #return df_m['M1 Excess Glucose'].iloc[int(index-1)] + (delta)
    else:
        return 0 #assumption-- start each day at 00:00:00 with no excess blood sugar
                 #this follows from the ADA guidelines (fasting blood sugar [more than
                 # 6 hours after last carbohydrate] of bsg under 126 mg/dL and between
                 # )
    
df_m["M1 Excess Glucose"] = df_m.apply(lambda row: model_glucose(row), axis =1)
df_m["M1 Excess Glucose"]= df_m["M1 Excess Glucose"].fillna(0)

df_m.iloc[40000:70000].plot(x = 'Abs Time', y= 'M1 Excess Glucose')
df_m.iloc[40000:70000].plot(x = 'Abs Time', y= 'Glucose from food (mg)')
#df['M1 dGlucose/dt'] = ['M1 Excess Glucose']*constant + df['Glucose from food']


# In[17]:


df_m[5630:5700]


# In[18]:


#Excess Glucose Stats
dates_in_sample = df["Date"].unique()
by_date = { date : df.loc[lambda df: df["Date"] == date,:].copy() for date in dates_in_sample}
fig, ax = plt.subplots(figsize = (15,10))

color_dict = {'Monday': 'b' ,'Tuesday': 'g' , 'Wednesday': 'r' , 'Thursday': 'c'}

dates = list(by_date.keys())

plt.title('Excess Blood Glucose Data : M-Th, {} through {}, n = {}'.format(dates[0],dates[-1],len(dates)))

times = ['09:00:00','12:00:00','17:00:00']

for date, series in by_date.items():
        day =series['Day of the week'].unique()[0]
        plt.plot(series['Abs Time'],series['Excess Glucose'] , color = color_dict[day], alpha =0.5)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off

plt.xlabel('Time of day- 00:00:00 through 24:00:00')

for time in times:
    x = int(time[:2])*3600
    plt.axvline(x = x, color = 'k', linestyle ='--')
    plt.text(x, 375, ' t= '+time)
    


