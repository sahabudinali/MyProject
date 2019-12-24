#!/usr/bin/env python
# coding: utf-8
#Content Based Recommendation System
# # IMPORT THE NEEDED PACKAGES

# In[3]:


import pandas as pd

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Now import the Dataset 

# In[4]:


movies_df=pd.read_csv("movies.csv")
ratings_df=pd.read_csv("ratings.csv")
movies_df.head(5)


# using regular expression to find a year sorted here we are trying to not conflict between the movies names and year

# In[6]:



movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)

movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)

movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()


# In[7]:


movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()


# Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
# 

# In[9]:


moviesWithGenres_df = movies_df.copy()


# For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
# 

# In[11]:


for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1

moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()


# In[12]:


ratings_df.head()


# Drop removes a specified row or column from a dataframe

# In[13]:


ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()


# In[14]:


userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies


# # Filtering out the movies by title

# In[15]:


inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]


# #Then merging it so we can get the movieId. It's implicitly merging it by title.

# In[16]:


inputMovies = pd.merge(inputId, inputMovies)


# Dropping information we won't use from the input dataframe
# 

# In[18]:


inputMovies = inputMovies.drop('genres', 1).drop('year', 1)


# Final Output Of The dataframe

# In[19]:


inputMovies


# Filtering the movie From the output

# In[20]:


userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies


# Resetting the index to avoid future issues

# In[21]:


userMovies = userMovies.reset_index(drop=True)


# Dropping unnecessary issues due to save memory and to avoid issues
# 

# In[22]:


userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
userGenreTable


# In[23]:


inputMovies['rating']


# Dot produt to get weights

# In[24]:


userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
userProfile


# Now let's get the genres of every movie in our original dataframe
# 

# In[25]:


genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])


# drop the unnecessary information

# In[26]:


genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()


# In[27]:


genreTable.shape


# Multiply the genres by the weights and then take the weighted average

# In[28]:


recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()


# Sort our Recommendation in Descending order

# In[30]:


recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
recommendationTable_df.head()


# Final Recommendation

# In[31]:


movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]


# In[ ]:




