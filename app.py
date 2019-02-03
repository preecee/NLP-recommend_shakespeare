#---general purpose libraries--
import numpy as np
import pandas as pd
import os
#--for text processing----
import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
stopwords = nltk.corpus.stopwords.words('english')

#***********************Generic Functions**************************************#

#----------------------------------------------------------------------------
#--- function to extract sonnets from the textfile and store in a dataframe--
#----------------------------------------------------------------------------
def build_sonnet_dataframe(file_name):
    corpus = open(file_name, encoding='utf-8')
    lines = corpus.readlines()
    sonnets = []
    paragraph = ''
    for line in lines:
        if line.isspace():
            if paragraph:
                sonnets.append(paragraph.strip())
                paragraph = ''
            continue
        else:
            if line.strip().isnumeric():
                continue
            else:
                paragraph += ' ' + line.strip()
    sonnets.remove(sonnets[0]) #--Remove heading---
    df = pd.DataFrame(data =sonnets, columns = ['Sonnet'] )
    return df

#----------------------------------------------------------------------------
#--- function to remove punctuation marks from a word--
#----------------------------------------------------------------------------
def remove_punctuation(word):
        punc_free= ''.join(ch for ch in word if ch not in string.punctuation)
        return punc_free

#----------------------------------------------------------------------------
#--- function to clean the corpus--
#----------------------------------------------------------------------------
def remove_common_n_stopwords(text):
    common = ['thy','thou','thee','shall','might']
    #Note: instead of using hardcoded common words, we can use 'Part of Speech tags from nltk to remove words like 'might'
    #however, for simplicity, I have just hardcoded the common words here.
    sen = ''
    for word in text.split():
        x= remove_punctuation(word)
        if x not in stopwords:
            if x not in common:
                sen+= ' ' + x
            else:
                continue
        else:
            continue
    return sen.strip()
#***********************Generic Functions ENDS here**********************************************************************************************#

#******************Gather the data from input sonnet file*****************************************************************************************
file_name = 'Shakespeare_Sonnets.txt'
#----call function to build the sonnet dataframe
df_sonnet = build_sonnet_dataframe(file_name)
v = df_sonnet['Sonnet'][0]
#************************Data gathered and stored in dataframe************************************************************************************#

#**********finding the most important, frequent word in each sonnet. Let it be 'Title' ***********************************************************#
#-----preprocess text data by removing stopwords,punctuation marks
df_sonnet['Sonnet_processed'] = df_sonnet['Sonnet'].apply(lambda x: remove_common_n_stopwords(x.lower()))

#--------------function to give the most important, frquent word in a sonnet----------#
def word_frequency_count(vec_obj,txt):
    vec = vec_obj.fit(txt)
    bag_of_words = vec.transform(txt)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq

#-----call function word_frequency_count() for all the sonnets and store the result in 'Title' column
df_sonnet['Title']=''
for index, row in df_sonnet.iterrows():
    words_freq=word_frequency_count(TfidfVectorizer(),row['Sonnet_processed'].split())
    #print(words_freq[0][0].upper())
    row['Title']=words_freq[0][0].upper()
df_sonnet.drop(columns=['Sonnet_processed'],inplace=True)
#************************'Title' stored in dataframe**************************************************************************************#

#**********************building sonnet recommender system**********************************************************************************
#---extracting keywords from all the sonnets in the dataframe
#--------------------------------------------------------------
df_sonnet['Keywords'] =''
for index, row in df_sonnet.iterrows():
    sonnet = row['Sonnet']
    #instantiate rake. This will remove all english stopwords and punctuation marks
    #--This could have been done using 'Sonnet_processed' column calculate above.
    #--However, I found better results by using Rake(). It also de-duplicates words
    rk= Rake()
    rk.extract_keywords_from_text(sonnet)
    keywords_scores = rk.get_word_degrees()
    # assigning the keywords to the new column for the corresponding sonnet
    Keywords= list(keywords_scores.keys())
    kywrd_clubbed=''
    for words in Keywords:
        kywrd_clubbed = kywrd_clubbed + ''.join(words)+ ' '
    row['Keywords'] = kywrd_clubbed
#-----------------------------------------------------------

#-----generate the count matrix---------
vec= CountVectorizer()
vec_matx=vec.fit_transform(df_sonnet['Keywords'])

#----- generating the cosine similarity matrix
cosine_sim = cosine_similarity(vec_matx, vec_matx)

#-- function to recommend 'n' number of sonnets based on an input sonnet
def sonnet_recommendation(choice):
    indx=pd.to_numeric(choice)-1
    recmnd_indx = list(pd.Series(cosine_sim[indx]).sort_values(ascending = False).index[:5])
    sonnet_recommendation = []
    sonnet_title = []
    sonnet_indx =[]
    print('Your recommended Shakespear sonnets are: ')
    for i in recmnd_indx:
        sonnet_recommendation.append(df_sonnet['Sonnet'][i])
        sonnet_title.append(df_sonnet['Title'][i])
        sonnet_indx.append(i+1)
    return sonnet_recommendation, sonnet_title, sonnet_indx
#**********************building sonnet recommender system ends here************************************************************************

#**********************Building the webservice ********************************************************************************************
#----------Import libraries required--------
from flask import Flask, request
from flask import render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
#----------libraries imported---------------

#--------setup the configuration for web form-------------
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
#------configuration done----------------------------------

#------create class to access flaskform------
class SearchForm(FlaskForm):
    index = StringField("Enter the index")
    submit = SubmitField("Search")
#----------class created----------------------

#--------create the routes----------------------------------------------------------------------------
@app.route('/recommend_shakespeare', methods=['GET','POST'])
def rec_sonnet():
    form = SearchForm()
    index = None
    rec_sonnet = []
    rec_title = []
    rec_indx = []
    if request.method == 'POST':
        index = form.index.data
        rec_sonnet, rec_title, rec_indx = sonnet_recommendation(index)
    return render_template('flask.html', form=form, index=index, rec_sonnet=rec_sonnet, rec_title=rec_title, rec_indx=rec_indx, name=v)
#------routes created-------------------------------------------------------------------------------------

#---------main function---------------------------
if __name__ == '__main__':
    app.run(debug=True)
#--------------------------------------------------
#********************Code ends here******************************************************************************************************************