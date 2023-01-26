#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import numpy as np
import pandas as pd 

from pandas import Series, DataFrame

# newly added in histogram
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# read data
data = pd.read_csv('DataScientist.csv')
# Before defining null values
data.head(5)


# In[2]:


# Define the null values as a list and pass it as an input while reading the data
missing_values = ["n/a", "na", "--", ' ', '?', 'NaN','-1']


# In[3]:


data = pd.read_csv('DataScientist.csv', na_values = missing_values)


# In[4]:


# after defining null values
data


# In[5]:


# Replace the 'null' values by average for the 'Rating' column
mean = data['Rating'].mean()
print('mean value=', mean)
print('Befor:\n ', data['Rating'])
data['Rating'].fillna(mean, inplace=True)
print('after:\n',data['Rating'])


# In[6]:


# after repacing null ratings
data


# In[7]:


# Drop Columns with missing data
missing_val_count_1 = data['Competitors'].isnull().sum()
missing_val_count_2 = data['Easy Apply'].isnull().sum()
rows = data['index'].count()
percentage_missing_1  =  (missing_val_count_1 *100)/rows
percentage_missing_2  =  (missing_val_count_2 *100)/rows
print('Percentage missing 1 = ', percentage_missing_1)
print('Percentage missing 2 = ', percentage_missing_2)
if percentage_missing_1 >=75.0:
    print('Delete Competitors coulmn')
    data.drop('Competitors', axis= 1,inplace = True)
if percentage_missing_2 >=75.0:
    print('Delete Easy Apply coulmn')
    data.drop('Easy Apply', axis= 1,inplace = True)  
print(data)


# In[8]:


# row count before data cleaning
rows = data['index'].count()
print('count =', rows)


# In[10]:


# Drop rows consisting of null or missing values
data.dropna(inplace=True)
data


# In[11]:


# row count after data cleaning
rows = data['index'].count()
print('row count =', rows)


# In[9]:


data.describe()


# In[10]:


# salary min/max estimation
hours_per_week = 40
weeks_per_year = 52

for i in range(data.shape[0]):
    salary_estimate = data.loc[i,"Salary Estimate"]
    salary_estimate = salary_estimate.replace("$", "")
    
    if "Per Hour" in salary_estimate:
        lower, upper = salary_estimate.split("-")
        upper, _ = upper.split("Per")
        upper= upper.strip()
        lower = int(lower) *hours_per_week*weeks_per_year*(1/1000)
        upper = int(upper) *hours_per_week*weeks_per_year*(1/1000)
        
    else:
        lower, upper = salary_estimate.split("-")
        lower = lower.replace("K", "")
        upper, _= upper.split("(")
        upper=upper.replace("K", "")
        upper = upper.strip()
    
        
    lower = int(lower)
    upper = int(upper)
    data.loc[i,"salary_estimate_lower_bound"] = lower
    data.loc[i,"salary_estimate_upper_bound"] = upper


# In[11]:


for i in range(data.shape[0]):
    name = data.loc[i,"Company Name"]
    if "\n" in name:
        name,_ = name.split("\n")
    data.loc[i,"Company Name"] = name


# In[12]:


data["Size"].value_counts()


# In[13]:


for i in range(data.shape[0]):
    size = data.loc[i,"Size"]
    if "to" in  size:
        lower,upper = size.split("to")
        lower = lower.strip() 
        _, upper, _ = upper.split(" ")
        upper = upper.strip()
        lower = int(lower)
        upper = int(upper)
    elif "+" in size:
        lower,_ = size.split("+")
        lower = int(lower)
        upper = np.inf
    else:
        lower = np.nan
        upper = np.nan
    data.loc[i,"Minimum Size"] = lower
    data.loc[i,"Maximum Size"] = upper


# In[21]:


data.head()


# In[22]:


data.drop(["Salary Estimate","Size"],axis=1,inplace=True)
data.head()


# In[4]:


# Print 'histogram' of the fonded year
plt.hist(data['Founded'], bins=2, rwidth=0.95)


# In[12]:


c_count = data['Job Title'].value_counts()['Data Analyst']
s_count = data['Job Title'].value_counts()['Data Scientist']

counts= [c_count, s_count]
labels = ['c', 's']
plt.pie(counts,  labels=labels, autopct='%1.2f%%')


# In[13]:


c_count = data['Job Title'].value_counts()['Data Analyst']
s_count = data['Job Title'].value_counts()['Data Scientist']

counts= [c_count, s_count]
labels = ['c', 's']
plt.bar(  labels, counts, color='pink')


# In[24]:


# best jobs by salary 
data[['Job Title','salary_estimate_upper_bound']].nlargest(10,"salary_estimate_upper_bound")


# In[25]:


# best jobs by company rating
data[['Job Title','Rating']].nlargest(10,"Rating")


# In[39]:


# top 20 companies in data science
plt.rcParams["figure.figsize"] = (10,10)
plt.style.use("seaborn")
color = plt.cm.viridis(np.linspace(0, 1, 15))
data["Company Name"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)
plt.title("Top 20 Company with Highest number of Jobs in Data Science",fontsize=20)
plt.xlabel("Company Name",fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[54]:


# top 20 jobs in data science
plt.rcParams['figure.figsize'] = (10,5)
color = plt.cm.viridis(np.linspace(0, 1, 256))
data["Job Title"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)
plt.title("Top 20 Data Science Job",fontsize=20)
plt.xlabel("Job Title",fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[52]:


top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 10)),
                       bottom(np.linspace(0, 1, 10))))

data["Headquarters"].value_counts().sort_values(ascending=False).head(20).plot.pie(y="Headquarters",colors=newcolors,autopct="%0.1f%%")
plt.title("Head Quarters according to Locations",fontsize=20)
plt.axis("off")
plt.show()


# In[15]:


from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'black', height =1000, width = 1000).generate(str(data["Industry"]))
plt.rcParams['figure.figsize'] = (10,10)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Most data science jobs available Industries")
plt.show()


# In[ ]:




