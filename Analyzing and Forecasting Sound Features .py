#!/usr/bin/env python
# coding: utf-8

# In this jupyter notebook I explore the changing sound features through the last 20 years and make predictions for the future. The sound features analyzed are:
# 
# **Danceability**:  Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.    
# **Speachiness**: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.   
# **Energy**: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.   
# **Instrumentalness**: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.  
# **Acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.    
# **Liveliness**: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.    
# **Valence**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).  
# 
# 

# In[1]:


import requests
import datetime
import base64
from urllib.parse import urlencode
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
get_ipython().run_line_magic('matplotlib', 'inline')
import csv
import numpy as np
from sklearn import metrics


# In[2]:


client_ID = 'a2bdd5ea206845ffb78af35ec64f4c67'
client_secret = 'f87660b748e24d468bd5b4ea1d18931a' 


# The spotify client was formed after whacting this video: https://www.youtube.com/watch?v=xdq6Gz33khQ. Thank you Coding Entrepreneurs for your spotify API tuturial. Note, https://github.com/codingforentrepreneurs/30-Days-of-Python/commit/9c8f45e840483133c340f10583ad6cb3f5e03fb6 is the orginial source code for the spotify api. I altered it based on my needs below. 

# In[3]:


class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    client_ID = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"
    
    def __init__(self, client_ID, client_secret, *args, **kwargs):
        #the super function helps avoid the usage of a parent class and  and enables multiple inheritances 
        super().__init__(*args,**kwargs) 
        self.client_ID = client_ID
        self.client_secret = client_secret
        
    def get_client_credentials(self):
        #Returns a base 64 encoded string
        client_ID = self.client_ID
        client_secret = self.client_secret
        if client_secret == None or client_ID == None:
            raise Exception("You must set client_id and client_secret")
        client_creds = f"{client_ID}:{client_secret}" #do not put a space between {client_ID}:{client_secret}, completely changes the code
        client_creds_b64 = base64.b64encode(client_creds.encode())
        return client_creds_b64.decode()

    def get_token_headers(self):
        client_creds_b64 = self.get_client_credentials()
        return {
            "Authorization" : f"Basic {client_creds_b64}" 
        }
    
    
    def get_token_data(self): 
        return {
            "grant_type": "client_credentials"
        }
    
    def perform_auth(self): 
        token_url = self.token_url
        token_data = self.get_token_data()
        token_headers = self.get_token_headers()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if r.status_code not in range(200, 299):
            raise Exception("Could not authenticate client.")
        data = r.json()
        now = datetime.datetime.now()
        access_token = data['access_token']
        expires_in = data['expires_in'] #seconds
        expires = now+datetime.timedelta(seconds = expires_in)
        self.access_token = access_token #grab the access token from the request (r.json) output
        self.access_token_expires = expires
        self.access_token_did_expire = expires < now
        return True 
    
    def get_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_access_token()
        elif token == None:
            self.perform_auth()
            return self.get_access_token()
        return token

    def get_resource_headers(self):
        access_token = self.get_access_token()
        
        #headers fields 
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        endpoint = "https://api.spotify.com/v1/search"
        return headers
    
 
    def base_search(self, query_params): # type
        headers = self.get_resource_headers()
        endpoint = "https://api.spotify.com/v1/search"
        lookup_url = f"{endpoint}?{query_params}"
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):  
            return {}
        return r.json()
    
    #ways to make search class better is add keyword matching and operator
    #idea below is to pass a dictionary or a string
    def search(self, query=None, operator=None, operator_query=None, search_type='artist' ):
        if query == None:
            raise Exception("A query is required")
        if isinstance(query, dict):
            query = " ".join([f"{k}:{v}" for k,v in query.items()])
        if operator != None and operator_query != None:
            if operator.lower == "or" or operator.lower == "not":
                operator = operator.upper()
                if isinstance(operator_query, str):
                    query = f"{query} {operator} {operator_query}"
        query_params = urlencode({"q": query, "type": search_type.lower()})
        print(query_params)
        return self.base_search(query_params)
    
    def sound_features(self, ids): 
        headers = self.get_resource_headers()
        endpoint = "https://api.spotify.com/v1/audio-features"
        data = urlencode({"ids":ids})
        lookup_url = f"{endpoint}?{data}"
        print(lookup_url)
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):  
            return {}
        return r.json()


# In[4]:


spotify = SpotifyAPI(client_ID,client_secret)


# Note, Everything below is my original code 

# In[1]:


#Takes in a list of ids and the year, grabs the audio feautres from the spotify api

def spotify_tool(track, Year):
    soundFeatures = []
    
    #much more than just audio feature 
    for x in track:
        search = spotify.search(x, search_type= "track")
        grub = search["tracks"]["items"][0]
        id1 = grub["id"]
        soundFeatures.append(spotify.sound_features(id1))
    
    #soundFeatures is a list of nested dictionaries, therefore nested for loop is neccesary 
    #Why did I name this info soundfeatures, that is so confusing 
    rows = []
    for x in soundFeatures:
        data_row = x["audio_features"]
        for row in data_row: 
            rows.append(row) 
    

    # create df that is easily downloadable for Tableau 
    df = pd.DataFrame(rows)
    df = df.drop(['type','uri','track_href','analysis_url','duration_ms','loudness','tempo','mode','id','time_signature','key'], axis=1)
    
    #change key to letters
    '''
    df['key'] = df['key'].replace([0],'C')
    df['key'] = df['key'].replace([1],'C#/Bb')
    df['key'] = df['key'].replace([2],'D')
    df['key'] = df['key'].replace([3],'D#/Eb')
    df['key'] = df['key'].replace([4],'E')
    df['key'] = df['key'].replace([5],'F')
    df['key'] = df['key'].replace([6],'F#/Gb')
    df['key'] = df['key'].replace([7],'G')
    df['key'] = df['key'].replace([8],'G#/Ab')
    df['key'] = df['key'].replace([9],'A')
    df['key'] = df['key'].replace([10],'A#/Bb')
    df['key'] = df['key'].replace([11],'B')
    
    '''
    
    df['Year']= Year
    
    #figure out how to automatically save this to Passion Project Folder
    #return df.to_csv('music_info.csv')
    return df


# The two codes below is how I extreacted the data from both Wikipedia and Billboard.com. Billboard.com does not have any data before 2006, therefore I had to resort to Wikipedia. My Wikipedia code will sometime error at 2006 however, there fore I keep the Billboard.com code.  
# Improvements I can make here are:  
# (1) Debug wikipedia to work for all years  
# (2) Include Artist in Search to make sure we are obtaining the right song

# In[55]:


years = ['2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015','2016','2017','2018','2019']
year_dict = {}
for x in years:
    page = requests.get(f"https://www.billboard.com/charts/year-end/{x}/hot-100-songs")
    if page.status_code != 200:
        print ("Page not downloaded successfully")
    soup = BeautifulSoup(page.content, 'html.parser')
    songs = []
    list_= soup.select('div[class = ye-chart-item__title]')
    for y in range(0,100):
        song = list_[y]
        songs.append(song.get_text().strip())
    year_dict[x] = songs


# In[58]:


years = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015','2016','2017','2018','2019']
year_dict = {}
for x in years:
    page = requests.get(f"https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{x}")
    if page.status_code != 200:
        print ("Page not downloaded successfully")
    soup = BeautifulSoup(page.content, 'html.parser')
    tables = soup.find_all('table', 'wikitable')
    rows = [row for row in tables[0].find_all('tr')]
    songs = []
    for y in range(1,101):
        try: 
            num_1 = rows[y].find_all('td')
            title = num_1[1].find('a').get('title')
        except:
            num_1 = [row for row in rows[y].find_all('td')]
            title = num_1[1].get_text()
        title = title.split('(', 1)[0]
        songs.append(title)
    #print (songs)
    year_dict[x] = songs


# The code  directly below should work. I need to correct it, but I have spent over a day trying to debug it now. That is why below if I just went ahead and called every year individually. I've deleted the code  for every year after 2000, where I ran each song individually because it took up a lot of space in the Notebook and looks messy. However, all I did was replace 2000 with 2001. 
# I think the error may have to deal with time? If I run two of the below functions at the same time it produces a key error.

# In[53]:


#### A huge data with 8 varibles(columns) and 2000 rows 
data = pd.DataFrame()
for key,values in year_dict.items():
    data.append(spotify_tool(values, key))


# In[10]:


zero = spotify_tool(year_dict['2000'], 2000)


# In the code below I saved the data so I don't have to run the codes above every time, I can just import a CSV file 

# In[ ]:


frames = [nineteen,eighteen,seventeen, sixteen, fifteen, fourteen, thirteen, twelve, eleven, ten, nine, eight, seven, six, five, four, three, two, one, zero]
results = pd.concat(frames)
results.to_csv(r'/Users/juliawilliams/Desktop/Spotify_Data.csv')


# In[2]:


results = pd.read_csv('Desktop_Spotify_Data.csv')
results


# ## ANALYSIS 
# In this section  I go through each of the sound feature and analyze what has happened over the past 20 years. Questions I asked my self while analyzing:  
# Have there been any features that have not changed?  
# What general trends do I see for each feature?  
# How do these trends reflect on society?  
# Music use to be about love, now its about sex and break up 

# ### Danceability 

# In[23]:


plt.figure(figsize=(16, 6))
sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
sns.set(style="ticks", color_codes=False)
sns.barplot(y = 'danceability',x = 'Year',palette="light:b",data = results)


# In[22]:


plt.figure(figsize=(16, 6))
sns.set_theme(style="darkgrid")
sns.lineplot(x="Year", y="danceability", data=results)


# ### Energy

# In[24]:


plt.figure(figsize=(16, 6))
sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
sns.set(style="ticks", color_codes=False)
sns.barplot(y = 'energy',x = 'Year',palette="light:b",data = results)


# In[25]:


plt.figure(figsize=(16, 6))
sns.lineplot(x="Year", y="energy", data=results)


# ### Speechiness
# Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

# In[26]:


plt.figure(figsize=(16, 6))
sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
sns.set(style="ticks", color_codes=False)
sns.barplot(y = 'speechiness',x = 'Year',palette="light:b",data = results)


# In[27]:


plt.figure(figsize=(16, 6))
sns.lineplot(x="Year", y="speechiness", data=results)


# ### Acousticness

# In[33]:


plt.figure(figsize=(16, 6))
sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
sns.set(style="ticks", color_codes=False)
sns.barplot(y = 'acousticness',x = 'Year',palette="light:b",data = results)


# In[28]:


plt.figure(figsize=(16, 6))
sns.lineplot(x="Year", y="acousticness", data=results)


# ### Instrumentalness

# In[34]:


plt.figure(figsize=(16, 6))
sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
sns.set(style="ticks", color_codes=False)
sns.barplot(y = 'instrumentalness',x = 'Year',palette="light:b",data = results)


# In[30]:


plt.figure(figsize=(16, 6))
sns.lineplot(x="Year", y="instrumentalness", data=results)


# ### Liveness
# 

# In[36]:


plt.figure(figsize=(16, 6))
sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
sns.set(style="ticks", color_codes=False)
sns.barplot(y = 'liveness',x = 'Year',palette="light:b",data = results)


# In[31]:


plt.figure(figsize=(16, 6))
sns.lineplot(x="Year", y="liveness", data=results)


# ### Valence
#  A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). 

# In[37]:


plt.figure(figsize=(16, 6))
sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
sns.set(style="ticks", color_codes=False)
sns.barplot(y = 'valence',x = 'Year',palette="light:b",data = results)


# In[32]:


plt.figure(figsize=(16, 6))
sns.lineplot(x="Year", y="valence", data=results)


# ## FORCASTING AND EVALUATING

# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ### Danceability 

# In[5]:


# Initialise and fit linear regression model using `statsmodels`
# Initialise and fit model
predictors =['Year']
X = results[predictors]
y = results['danceability']
lm = LinearRegression()
model = lm.fit(X,y)


# In[6]:


print(f'alpha = {model.intercept_}')
print(f'betas = {model.coef_}')


# DANCIBILITY = YEAR*3.08+.00186404

# In[7]:


dance_pred = model.predict(X)


# In[8]:


# Plot regression against actual data
#sns.figure(figsize=(12, 6))
plt.figure(figsize=(16, 6))
sns.lineplot(y = 'danceability',x = 'Year', data = results)           # scatter plot showing actual data
sns.lineplot(y = dance_pred,x = 'Year', data = results).set(xlabel='Year', ylabel='Danceability')


# Since i used prediction data to test this is not representative of anything 

# In[9]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y, dance_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y, dance_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, dance_pred)))


# In[53]:


Xnew = [[2020]]
model.predict(Xnew)


# ### Engergy

# In[10]:


predictors =['Year']
X = results[predictors]
y = results['energy']
lm = LinearRegression()
model = lm.fit(X,y)

#equation
print(f'alpha = {model.intercept_}')
print(f'betas = {model.coef_}')
energy_pred = model.predict(X)

# Plot regression against actual data
#sns.figure(figsize=(12, 6))
plt.figure(figsize=(16, 6))
sns.lineplot(y = 'energy',x = 'Year', data = results)           # scatter plot showing actual data
sns.lineplot(y = energy_pred,x = 'Year', data = results).set(xlabel='Year', ylabel='Energy')


# ### Valence

# In[20]:



predictors =['Year']
X = results[predictors]
y = results['valence']
lm = LinearRegression()
model = lm.fit(X,y)


# In[21]:


print(f'alpha = {model.intercept_}')
print(f'betas = {model.coef_}')
val_pred = model.predict(X)
# Plot regression against actual data
#sns.figure(figsize=(12, 6))
plt.figure(figsize=(16, 6))
sns.lineplot(y = 'valence',x = 'Year', data = results)           # scatter plot showing actual data
sns.lineplot(y = val_pred,x = 'Year', data = results).set(xlabel='Year', ylabel='valence')


# 

# In[ ]:




