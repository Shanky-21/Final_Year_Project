#!/usr/bin/env python
# coding: utf-8

# # RESTAURANT RECOMMENDER SYSTEM
# 
# This data set consists of restaurants of Hyderabad/India collected from Zomato.
# 
# My aim is to create a content based recommender system in which;
# * I will write a restaurant name,
# * Recommender system will look at the reviews of other restaurants
# * System will recommend us other restaurants with similar reviews and sort them from the highest rated.

# In[1]:

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:

st.title("Food Recommender System")

data_names = pd.read_csv('Restaurant names and Metadata.csv')
data_reviews = pd.read_csv('Restaurant reviews.csv')


# In[3]:


data_names.head()


# In[4]:


data_reviews.head()


# In[5]:


# Looking at the data types and NaN values
data_names.info()


# In[6]:


# Counting the unique values
data_names.nunique()


# In[7]:


# Looking at the data types and NaN values
data_reviews.info()


# In[8]:


# Counting the unique values
data_reviews.nunique()


# ## Merging Two Data Sets
# 
# I will merge these two data sets.
# 
# After the merging I will have a data set with individual customer reviews and ratings for the restaurants.

# In[9]:

# Renaming the restaurant name column with the same value as in the other data set:
data_reviews = data_reviews.rename(columns={'Restaurant': 'Name'})

# Merging the two data sets:
df = pd.merge(data_reviews, data_names, how='left', on='Name')

# Dropping the columns which I am not going to use:
df.drop(['Reviewer', 'Time', 'Pictures', 'Links', 'Collections'], axis=1, inplace=True)
df.head()


# ## Preparing Cost and Rating Columns

# In[10]:


# Changing cost and rating columns data types:
df['Cost'] = df['Cost'].str.replace(',', '').astype(int)
df['Rating'] = df['Rating'].str.replace('Like', '1').astype(float)
df.info()


# ## Handling Missing Values

# In[11]:


print('Nu of data inputs:', len(df))
print('\nNu of NaN values for each column:\n')
print(df.isnull().sum())


# Rating column is important for recommender system. So I am not going to drop those 38 NaN values.
# 
# Let's examine data with NaN rating value.  
# 
# I will fill those NaN values with each restaurants' mean rating value.

# In[12]:


# Examine missing Rating values:
df['Name'][df['Rating'].isnull() == True].value_counts()


# So there are only two restaurants with total of 38 NaN rating values.
# 
# Let's see individual restaurant's average rating value.

# In[13]:


print('Mean of Rating for American Wild Wings: ', df['Rating'][df['Name'] == 'American Wild Wings'].mean())
print('Mean of Rating for Arena Eleven: ', df['Rating'][df['Name'] == 'Arena Eleven'].mean())
print('Overall Mean of Ratings: ', df['Rating'].mean())


# We can see that mean value for missing rating value should be 4 (3,9 and 4,1 for each restaurant). 
# 
# Let's fill those restaurants missing rating values.

# In[14]:


df['Rating'].fillna(4, inplace=True)

# Changing NaN reviews by '-'
df['Review'] = df['Review'].fillna('-')
df.isnull().sum()


# ## Separating Metadata (Reviews and Followers)
# 
# I will separate review and follower numbers into different columns in order to use it later.

# In[15]:


# Filling missing values:
df['Metadata'].fillna('0 Review , 0 Follower', inplace=True)

# Standardizing strings
df['Metadata'] = df['Metadata'].str.replace('Reviews', 'Review')
df['Metadata'] = df['Metadata'].str.replace('Followers', 'Follower')

df['Metadata'][df['Metadata'].str.endswith('w')] = df['Metadata'][df['Metadata'].str.endswith('w')] + ' , - Follower'

# Splitting into two columns
df[['Reviews', 'Followers']] = df['Metadata'].str.split(' , ', expand=True)

# Erasing wording from the columns
df['Reviews'] = df['Reviews'].str.replace('Review', '')
df['Reviews'] = df['Reviews'].str.replace('Posts', '')
df['Reviews'] = df['Reviews'].str.replace('Post', '')

df['Followers'] = df['Followers'].str.replace('Follower', '')
df['Followers'] = df['Followers'].str.replace('-', '0')

# Changing str values to integers
df[['Reviews', 'Followers']] = df[['Reviews', 'Followers']].astype(int)

# Dropping the initial column
df.drop(['Metadata'], axis=1, inplace=True)

# Sorting restaurants with their names and costs
df = df.sort_values(['Name', 'Cost'], ascending=False).reset_index()
df.drop('index', axis=1, inplace=True)


# In[16]:

#df.head()


# ## Creating New Features (Mean of Ratings, Reviews, and Followers)
# 
# Rating, Review, and Followers columns represents individual customers' inputs.
# 
# I am going to find the means of these values and assign them for the restaurants.

# In[17]:


restaurants = list(df['Name'].unique())
df['Mean Rating'] = 0
df['Mean Reviews'] = 0
df['Mean Followers'] = 0

for i in range(len(restaurants)):
    df['Mean Rating'][df['Name'] == restaurants[i]] = df['Rating'][df['Name'] == restaurants[i]].mean()
    df['Mean Reviews'][df['Name'] == restaurants[i]] = df['Reviews'][df['Name'] == restaurants[i]].mean()
    df['Mean Followers'][df['Name'] == restaurants[i]] = df['Followers'][df['Name'] == restaurants[i]].mean()


# In[18]:


df.sample(3)


# ## Feature Scaling
# 
# I will scale the features between 1-5.

# In[19]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (1,5))

df[['Mean Rating', 'Mean Reviews', 'Mean Followers']] = scaler.fit_transform(df[['Mean Rating', 'Mean Reviews', 'Mean Followers']]).round(2)

df.sample(3)


# ## Text Preprocessig and Cleaning
# 
# We will be using 'Review' and 'Cuisines' feature'in order to create a recommender system.
# 
# So we need to prepare and clean the text in those columns.

# In[20]:


import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[21]:


# 5 examples of these columns before text processing:
df[['Review', 'Cuisines']].sample(5)


# In[22]:


# Define symbols to be replaced by space
replace_space = re.compile('[/(){}\[\]\|@,;]')
# Define symbols to be removed
remove_symbols = re.compile('[^0-9a-z #+_]')
# Define stopwords
stopwords = set(stopwords.words('english'))

def text_preprocessing(text):
    # Lowercase all the letters
    text = text.lower()
    
    # Replace these symbols with space
    text = replace_space.sub(' ', text)
    
    # Remove these symbols
    text = remove_symbols.sub('', text)
    
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stopwords)
    
    return text


# In[23]:


df['Review'] = df['Review'].apply(text_preprocessing)
df['Cuisines'] = df['Cuisines'].apply(text_preprocessing)


# In[24]:


# Columns after processed:
df[['Review','Cuisines']].sample(5)


# ## EDA - Analysing Restaurants and Popularities

# In[25]:


# RESTAURANT NAMES:
restaurant_names = list(df['Name'].unique())
#restaurant_names


# In[26]:


df_rating = df.drop_duplicates(subset='Name')
df_rating = df_rating.sort_values(by='Mean Rating', ascending=False).head(10)

plt.figure(figsize=(7,5))
sns.barplot(data=df_rating, x='Mean Rating', y='Name', palette='RdBu')
plt.title('Top Rated 10 Restaurants');


# In[27]:


df_reviews = df.drop_duplicates(subset='Name')
df_reviews = df_reviews.sort_values(by='Mean Reviews', ascending=False).head(10)

plt.figure(figsize=(7,5))
sns.barplot(data=df_reviews, x='Mean Reviews', y='Name', palette='RdBu')
plt.title('Top Reviewed 10 Restaurants');


# In[28]:

df_followers = df.drop_duplicates(subset='Name')
df_followers = df_followers.sort_values(by='Mean Followers', ascending=False).head(10)

plt.figure(figsize=(7,5))
sns.barplot(data=df_followers, x='Mean Followers', y='Name', palette='RdBu')
plt.title('Most Followed Top 10 Restaurants');


# ## EDA - Word Frequency Distribution:

# In[29]:


def get_top_words(column, top_nu_of_words, nu_of_word):
    
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    
    bag_of_words = vec.fit_transform(column)
    
    sum_words = bag_of_words.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:top_nu_of_words]


# In[30]:


# Top 20 two word frequencies for Cuisines
list1 = get_top_words(df['Cuisines'], 20, (2,2))

df_words1 = pd.DataFrame(list1, columns=['Word', 'Count'])

plt.figure(figsize=(7,6))
sns.barplot(data=df_words1, x='Count', y='Word')
plt.title('Word Couple Frequency for Cuisines');


# In[31]:


# Top 20 two word frequencies for Reviews
list2 = get_top_words(df['Review'], 20, (2,2))

df_words2 = pd.DataFrame(list2, columns=['Word', 'Count'])

plt.figure(figsize=(7,6))
sns.barplot(data=df_words2, x='Count', y='Word')
plt.title('Word Couple Frequency for Reviews');


# # CONTENT BASE RECOMMENDER SYSTEM
# 
# ## TF-IDF Matrix (Term Frequency — Inverse Document Frequency Matrix)
# 
# TF-IDF method is used to quantify words and compute weights for them. 
# 
# In other words, representing each word (or couples of words etc.) with a number in order to use mathematics in our recommender system.
# 
# Cosine similarity is a metric used to determine how similar the documents are irrespective of their size.

# In[32]:


# Changing data set index by restaurant name
df.set_index('Name', inplace=True)

# Saving indexes in a series
indices = pd.Series(df.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Review'])

# Calculating cosine similarities
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


# ## Creating the Recommender System:

# In[33]:


def recommend(name, cosine_similarities = cosine_similarities):
    
    # Create a list to put top 10 restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df.index)[each])
    
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['Cuisines', 'Mean Rating', 'Cost', 'Timings'])
    
    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df[['Cuisines','Mean Rating', 'Cost', 'Timings']][df.index == each].sample()))
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['Cuisines','Mean Rating', 'Cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    #print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new


# ## Testing the Recommender System
# 
# ## 1. Example:

# In[34]:


# HERE IS A RANDOM RESTAURANT. LET'S SEE THE DETAILS ABOUT THIS RESTAURANT:
df[df.index == 'Hyderabadi Daawat'].head(1)


# In[35]:


# LET'S SEE WHAT ARE WE GOING TO BE RECOMMENDED:
#recommend('Hyderabadi Daawat')


# ## 2. Example:

# In[36]:


# HERE IS A BAKERY. LET'S SEE THE DETAILS ABOUT THIS RESTAURANT:
df[df.index == 'Labonel'].head(1)


# In[37]:


# LET'S SEE WHAT ARE WE GOING TO BE RECOMMENDED:
#recommend('Labonel')


# ## 3. Example:

# In[38]:


# HERE IS A MEDITERRANEAN / NORT INDIAN / KEBAB / BBQ RESTAURANT. LET'S SEE THE DETAILS ABOUT THIS RESTAURANT:
df[df.index == 'Barbeque Nation'].sample(1)


# In[39]:


# LET'S SEE WHAT ARE WE GOING TO BE RECOMMENDED:
    
user_input = st.text_input("Enter the Name of Restaurant", 'Barbeque Nation')

df = pd.DataFrame(recommend(user_input))
st.write(df)


# Thanks for your attention and please upvote if you appreciate my work.
# 
# Melih
