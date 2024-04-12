#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# In[2]:


demographics = pd.read_csv('/Users/linus/Documents/CA259/demographics.csv')
demographics


# In[3]:


demographics.isna().sum()
#cheking to make sure we have the right number of missing values 


# In[4]:


demographics.describe()# see number of younger and number of older siblings the mean is quite different


# In[51]:


demographics_is_na = demographics.iloc[:5,:].copy()
demographics_is_na


# In[10]:


demographics_no_na = demographics.iloc[5:,:].copy()
demographics_no_na.isna().sum()


# In[15]:


demographics_no_na_post_dist = demographics_no_na[['Daily travel to DCU (in km, 0 if on-campus)', 'Old Dublin postcode (0 if outside Dublin)']]
demographics_no_na_post_dist.groupby('Old Dublin postcode (0 if outside Dublin)').mean()


# In[18]:


demographics_no_na_post_dist.groupby('Old Dublin postcode (0 if outside Dublin)').std()


# In[19]:


demographics_no_na_post_dist.groupby('Old Dublin postcode (0 if outside Dublin)').max()


# In[52]:


demographics_is_na.iloc[1,2] = 11.60 #even though the mean for a non-existing post code makes no sense, we use it, because with a real dataset this would be the case
demographics_is_na


# In[32]:


demographics_no_na['Gender'].value_counts()


# In[41]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_no_na[['Age (in years)','Average year 1 exam result (as %)']]
yDf = demographics_no_na['CAO Points (100 to 600)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)
print(reg_lin.predict([[21,66]]))


# In[36]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_no_na[['Average year 1 exam result (as %)']]
yDf = demographics_no_na['CAO Points (100 to 600)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))



# In[38]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_no_na[['Age (in years)','Average year 1 exam result (as %)','Height (in cm)','Weight (in kg)','Seat row in class','Daily travel to DCU (in km, 0 if on-campus)','Number of older siblings','Number of younger siblings','Old Dublin postcode (0 if outside Dublin)','Shoe size']]
yDf = demographics_no_na['CAO Points (100 to 600)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))


# In[53]:


demographics_is_na.iloc[0,1] = 491
demographics_is_na


# In[43]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_no_na[['Age (in years)','CAO Points (100 to 600)']]
yDf = demographics_no_na['Average year 1 exam result (as %)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)
print(reg_lin.predict([[20,600]]))


# In[54]:


demographics_is_na.iloc[2,3] = 70.5
demographics_is_na


# In[46]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_no_na[['Age (in years)','CAO Points (100 to 600)','Average year 1 exam result (as %)']]
yDf = demographics_no_na['Seat row in class']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)
print(reg_lin.predict([[19,543,71]]))


# In[49]:


#no meaningful correlation, which was not expected in any other way
#therefore we take a random number
np.random.randint(1,12)


# In[55]:


demographics_is_na.iloc[3,4] = 10
demographics_is_na


# In[56]:


#because we can not predict the age here we use the mean to fill it in
demographics_is_na.iloc[4,0] = 22
demographics_is_na


# In[59]:


normalized_demographics = pd.concat([demographics_is_na,demographics_no_na], axis = 0)
normalized_demographics


# In[60]:


personalities = pd.read_csv('/Users/linus/Documents/CA259/personalities.csv')
personalities


# In[65]:


personalities.rename(columns={'Last 4 digits of your mobile (same as on previous form)':'Last 4 digits of your mobile (0000 to 9999)',
                             'Your rating for EXTRAVERSION (vs. introversion)' :'EXTRAVERSION',
                             'Your rating for INTUITION (vs. observation)': 'INTUITION',
                             'Your rating for THINKING (vs. feeling)': 'THINKING',
                             'Your rating for JUDGING (vs. prospecting)': 'JUDGING',
                             'Your rating for ASSERTIVE (vs. turbulent)':'ASSERTIVE'}, inplace = True)


# In[66]:


personalities[personalities['Last 4 digits of your mobile (0000 to 9999)'] == 4397]


# In[90]:


personalities[personalities['Last 4 digits of your mobile (0000 to 9999)'].duplicated()]


# In[67]:


personalities.drop(53, axis=0, inplace = True)


# In[69]:


personalities[personalities['Last 4 digits of your mobile (0000 to 9999)'] == 4397]


# In[74]:


demographics_personalities = normalized_demographics.merge(personalities, on = 'Last 4 digits of your mobile (0000 to 9999)', how = 'inner')
demographics_personalities


# In[73]:


demographics_personalities.isna().sum()


# In[83]:


common_lst = []
for el in normalized_demographics['Last 4 digits of your mobile (0000 to 9999)'].unique():
    if el in personalities['Last 4 digits of your mobile (0000 to 9999)'].unique():
        common_lst.append(el)
print(len(common_lst))                         


# In[84]:


demographics_personalities['Last 4 digits of your mobile (0000 to 9999)'].unique()


# In[86]:


outer = normalized_demographics.merge(personalities, on = 'Last 4 digits of your mobile (0000 to 9999)', how = 'outer')
outer['Last 4 digits of your mobile (0000 to 9999)'].unique()


# In[91]:


personalities.drop(89, axis=0, inplace = True)
personalities.drop(97, axis=0, inplace = True)
personalities


# In[92]:


demographics_personalities = normalized_demographics.merge(personalities, on = 'Last 4 digits of your mobile (0000 to 9999)', how = 'inner')
demographics_personalities


# In[99]:


demographics_personalities[['Gender','EXTRAVERSION','INTUITION','THINKING','JUDGING','ASSERTIVE']].groupby('Gender').mean()


# In[102]:


men_women_pers_data = demographics_personalities[['Gender','EXTRAVERSION','INTUITION','THINKING','JUDGING','ASSERTIVE']]
men_pers_data = men_women_pers_data[men_women_pers_data['Gender']== 'Male']
women_pers_data = men_women_pers_data[men_women_pers_data['Gender']== 'Female']

men_pers_data


# In[105]:


from scipy import stats

# Sample data
men_extra = men_pers_data['EXTRAVERSION']
women_extra = women_pers_data['EXTRAVERSION']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_extra, women_extra)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[107]:


# Sample data
men_int = men_pers_data['INTUITION']
women_int = women_pers_data['INTUITION']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_int, women_int)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[108]:


# Sample data
men_th = men_pers_data['THINKING']
women_th = women_pers_data['THINKING']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_th, women_th)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[109]:


# Sample data
men_jd = men_pers_data['JUDGING']
women_jd = women_pers_data['JUDGING']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_jd, women_jd)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[110]:


# Sample data
men_ass = men_pers_data['ASSERTIVE']
women_ass = women_pers_data['ASSERTIVE']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(men_ass, women_ass)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[112]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_personalities[['Weight (in kg)','Shoe size']]
yDf = demographics_personalities['Height (in cm)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)
#print(reg_lin.predict([[19,543,71]]))


# In[136]:


demographics_personalities['total_siblings'] = demographics_personalities['Number of older siblings'] + demographics_personalities['Number of younger siblings']
demographics_personalities


# In[137]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_personalities[['total_siblings']]
yDf = demographics_personalities['CAO Points (100 to 600)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)


# In[117]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_personalities[['Number of older siblings', 'Number of younger siblings']]
yDf = demographics_personalities['Average year 1 exam result (as %)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)


# In[118]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_personalities[['EXTRAVERSION','INTUITION','THINKING','JUDGING','ASSERTIVE']]
yDf = demographics_personalities['Average year 1 exam result (as %)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)


# In[119]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_personalities[['EXTRAVERSION','INTUITION','THINKING','JUDGING','ASSERTIVE']]
yDf = demographics_personalities['CAO Points (100 to 600)']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)


# In[138]:


correlations = demographics_personalities.drop(columns = ['Gender', 'Hair colour', 'Star sign', 'Eye colour']).corr()
correlations


# In[185]:


tpllst = []
for i in range(0,18):
    for j in range(0,18):
        if correlations.iloc[i,j] > 0.6 and correlations.iloc[i,j] < 1.0:
          tpllst.append((correlations.index[i], correlations.columns[j], correlations.iloc[i, j]))
tpllst


# In[144]:


(correlations.iloc[2,*]).index()


# In[147]:


print(demographics_personalities['Hair colour'].unique())
print(demographics_personalities['Eye colour'].unique())
# we can use label encoding if we assume that we have ordinal ranking for lighter to darker hair, i.e. blond is the most light and bacl is the most dark
# we can use the same for the eye color


# In[148]:


encode_hair_dict = {'Blonde' : 0,
                   'Red' : 1,
                   'Brown' : 2,
                   'Black' : 3}
demographics_personalities['Hair colour encoded'] = demographics_personalities['Hair colour'].map(encode_hair_dict)
demographics_personalities


# In[150]:


encode_eye_dict = {'Blue' : 0,
                   'Green' : 1,
                   'Brown' : 2}
demographics_personalities['Eye colour encoded'] = demographics_personalities['Eye colour'].map(encode_eye_dict)
demographics_personalities


# In[151]:


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = demographics_personalities[['Hair colour encoded']]
yDf = demographics_personalities['Eye colour encoded']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin.score(X_test, y_test))
print(reg_lin.coef_)


# In[152]:


demographics_personalities[['Hair colour encoded','Eye colour encoded']].corr()


# In[154]:


demographics_personalities[['Star sign','EXTRAVERSION','INTUITION','THINKING','JUDGING','ASSERTIVE']]
.groupby('Star sign').mean()


# In[155]:


demographics_personalities['Star sign'].value_counts()


# In[159]:


taur_extra = demographics_personalities[demographics_personalities['Star sign'] == 'Taurus']['EXTRAVERSION']
taur_extra


# In[160]:


taur_extra = demographics_personalities[demographics_personalities['Star sign'] == 'Taurus']['EXTRAVERSION']
libra_extra = demographics_personalities[demographics_personalities['Star sign'] == 'Libra']['EXTRAVERSION']

t_statistic, p_value = stats.ttest_ind(taur_extra, libra_extra)

print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[161]:


# Sample data
pices_int = demographics_personalities[demographics_personalities['Star sign'] == 'Pices']['INTUITION']
libra_int = demographics_personalities[demographics_personalities['Star sign'] == 'Libra']['INTUITION']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(pices_int, libra_int)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[163]:


# Sample data
gemini_th = demographics_personalities[demographics_personalities['Star sign'] == 'Gemini']['THINKING']
saggi_th = demographics_personalities[demographics_personalities['Star sign'] == 'Sagittarius']['THINKING']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(gemini_th, saggi_th)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[164]:


# Sample data
cap_jd = demographics_personalities[demographics_personalities['Star sign'] == 'Capricorn']['JUDGING']
libra_jd = demographics_personalities[demographics_personalities['Star sign'] == 'Libra']['JUDGING']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(cap_jd, libra_jd)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[165]:


# Sample data
sagg_ass = demographics_personalities[demographics_personalities['Star sign'] == 'Sagittarius']['ASSERTIVE']
virgo_ass = demographics_personalities[demographics_personalities['Star sign'] == 'Virgo']['ASSERTIVE']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(sagg_ass, virgo_ass)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[184]:


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot the first line graph on the first subplot
ax1.plot(range(0,10), taur_extra, color='blue', label='Taurus')
ax1.set_ylabel('Personality Score')
ax1.set_title('Taurus and Libra extraversion')
ax1.legend()
ax1.set_ylim(0,100)
ax1.set_xticks([])

# Plot the second line graph on the second subplot
ax2.plot(range(0,5), libra_extra, color='red', label='Libra')
ax2.set_ylabel('Y')
ax2.set_title('Line Graph 2')
ax2.legend()
ax2.set_ylim(0,100)
ax2.set_xticks([])


# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# In[183]:


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot the first line graph on the first subplot
ax1.plot(range(0,7), gemini_th, color='blue', label='Gemini')
ax1.set_ylabel('Personality Score')
ax1.set_title('Gemini and Sagittarius Thinking')
ax1.legend()
ax1.set_ylim(0,100)
ax1.set_xticks([])

# Plot the second line graph on the second subplot
ax2.plot(range(0,6), saggi_th, color='red', label='Sagittarius')
ax2.set_ylabel('Y')
ax2.set_title('Line Graph 2')
ax2.legend()
ax2.set_ylim(0,100)
ax2.set_xticks([])


# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# In[ ]:




