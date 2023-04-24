#!/usr/bin/env python
# coding: utf-8

# # Importing  Required Libraries

# In[23]:


import os
import sys
import time
import wave

from IPython.display import Audio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import librosa
import librosa.display

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# # Loading the audio files

# In[2]:


audio_dir="/Users/kanika2601/Desktop/ALL"


# In[3]:


audio_files=[os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]


# In[38]:


audio_files


# In[4]:


len(audio_files)


# In[5]:


l1=[]
l2=[]
l3=[]
l4=[]
for i in range(len(audio_files)):
    file_name=audio_files[i].split("/")[-1].split("_")[-1]
    audio_path = audio_files[i]
    y, sr = librosa.load(audio_path)
    l3.append(y)
    l4.append(sr)
    if file_name[0]=="n":#neutral
        l1.append(audio_files[i].split("/")[-1])
        l2.append("neutral")
    elif file_name[0] == 'h':#happy
        l1.append(audio_files[i].split("/")[-1])
        l2.append("happy")
    elif file_name[0] == 'd':#disgust
        l1.append(audio_files[i].split("/")[-1])
        l2.append("disgust")
    elif file_name[0] == 'a':#angry
        l1.append(audio_files[i].split("/")[-1])
        l2.append("angry")
    elif file_name[0] == 'c':#calm
        l1.append(audio_files[i].split("/")[-1])
        l2.append("calm")
    elif file_name[0] == 'f':#fearful
        l1.append(audio_files[i].split("/")[-1])
        l2.append("fearful")
    elif file_name[0] == 's':
        if file_name[1] == 'a':#sad
            l1.append(audio_files[i].split("/")[-1])
            l2.append("sad")
        elif file_name[1] == 'u':#surprise
            l1.append(audio_files[i].split("/")[-1])
            l2.append("surprise")


# In[6]:


list_emotions = list(zip(l1,l2))  
audio_array = list(zip(l3,l2))  
list_paths = list(zip(audio_files,l2)) 


# In[7]:


l3=np.array(l3)


# In[8]:


list_emotions[0][1]


# In[9]:


len(l3)


# In[10]:


hop_length = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)

plt.figure(figsize=(12,8))
for i in range(20):
    plt.plot(tempogram[i], label=i)
plt.legend()
plt.title("Tempogram")
plt.show()


# # Genrating various DataFrames for later use

# In[11]:


df_path=pd.DataFrame(list_paths, columns=['File_Path', 'Emotion_label'])


# In[12]:


df_path.head()


# In[13]:


df_files=pd.DataFrame(list_emotions, columns=['File_Name', 'Emotion_label'])


# In[14]:


df_files.head()


# # Exploratory Data Analysis

# In[44]:


sns.countplot(x='Emotion_label',data=df_emotions)


# In[45]:


fig = px.sunburst(df_emotions, path=['Emotion_label','File_Name'])
fig.show()


# In[ ]:





# In[46]:


df_audioArray=pd.DataFrame(audio_array,columns=['Audio_Array', 'Emotion_label'])


# In[47]:


df_audioArray.head()


# # Some operations on Audio Files

# In[48]:


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data):
    return librosa.effects.time_stretch(data,rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr = sampling_rate, n_steps = pitch_factor)


# # Feature Extraction using MFCC

# In[49]:


def extract_features(x, sr):
    result = np.array([])
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc)) 
    return result


# In[50]:


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
  
    
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result


# # Getting Final DataFrame

# In[51]:


X, Y = [], []
for path, emotion in zip(df_path.File_Path, df_path.Emotion_label):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        
        Y.append(emotion)


# In[52]:


Emotions = pd.DataFrame(X)
Emotions['labels'] = Y
Emotions.head()


# # Data Preprocessing 

# In[24]:


X = Emotions.iloc[: ,:-1].values
Y = Emotions['labels'].values


# In[25]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# In[26]:


encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[28]:


X_train = x_train.reshape(x_train.shape[0] , x_train.shape[1] , 1)
X_test = x_test.reshape(x_test.shape[0] , x_test.shape[1] , 1)


# In[29]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# # Decision Tree

# In[31]:


from sklearn.tree import DecisionTreeClassifier
clf3 = DecisionTreeClassifier()

clf3 = clf3.fit(x_train,y_train)

y_pred = clf3.predict(x_test)


# In[32]:


print("Training set score: {:.3f}".format(clf3.score(x_train, y_train)))
print("Test set score: {:.3f}".format(clf3.score(x_test, y_test)))


# # KNN

# In[33]:


from sklearn.neighbors import KNeighborsClassifier
clf1=KNeighborsClassifier(n_neighbors=4)
clf1.fit(x_train,y_train)


# In[34]:


y_pred=clf1.predict(x_test)
print("Training set score: {:.3f}".format(clf1.score(x_train, y_train)))
print("Test set score: {:.3f}".format(clf1.score(x_test, y_test)))


# # Multi Layer Perceptron Model

# In[36]:


from sklearn.neural_network import MLPClassifier
clf2=MLPClassifier(alpha=0.01, batch_size=270, epsilon=1e-08, hidden_layer_sizes=(400,), learning_rate='adaptive', max_iter=400)
clf2.fit(x_train,y_train)


# In[37]:


print("Training set score: {:.3f}".format(clf2.score(x_train, y_train)))
print("Test set score: {:.3f}".format(clf2.score(x_test, y_test)))


# # Random Forest

# In[62]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[63]:


rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rfc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rfc.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[ ]:





# In[ ]:





# In[ ]:




