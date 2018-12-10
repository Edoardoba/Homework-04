
# coding: utf-8

# In[ ]:



# coding: utf-8

# In[ ]:


import time
from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
import math
from sklearn.cluster import KMeans
import collections
from collections import defaultdict
import os

def web_scrap(): 
    print ("Start : %s" % time.ctime())  #Just to see the time
    url="https://www.immobiliare.it/vendita-case/roma/?pag="
    counturl=1
    ndocs=int(14000/25)
    titles=[]
    descriptions=[]
    prices=[]
    locali=[]
    sup=[]
    bath=[]
    piano=[]
    while counturl<=ndocs:
        housesites=[]
        goodurl=url+str(counturl)
        page = requests.get(goodurl)
        soup = BeautifulSoup(page.content, 'html.parser')
        for a in soup.find_all('a', href=True,title=True,id=True):  #With this I get all the links of the page corresponding to houses
            housesites.append(a['href'])
        for a in housesites:
            time.sleep( 0.005 )
            try:
                page = requests.get(a)
                soup = BeautifulSoup(page.content, 'html.parser')
                check=soup.find_all('span',attrs = {'class':'text-bold'})[0].get_text() 
                if "€" in check:
                    a=soup.find_all('h1',attrs = {'class':'raleway title-detail'})[0].get_text()
                    if "asta" in a:
                        next
                    else:
                        b=soup.find_all('div',attrs = {'class':'col-xs-12 description-text text-compressed'})[0].get_text()
                        c=soup.find_all('li' ,attrs={'class':'features__price'})[0].get_text()
                        d=soup.find_all('span',attrs = {'class':'text-bold'})[1].get_text() 
                        e=soup.find_all('span',attrs = {'class':'text-bold'})[2].get_text()
                        f=soup.find_all('span',attrs = {'class':'text-bold'})[3].get_text()
                        g=soup.find_all('abbr',attrs = {'class':'text-bold im-abbr'})[0].get_text()
                        c=c.replace("€","")
                        g=g.replace("\n","")
                        b=b.replace("\n","")
                        g=g.replace("T","0")
                        titles.append(a)
                        descriptions.append(b)
                        prices.append(c)
                        locali.append(d)
                        sup.append(e)
                        bath.append(f)
                        piano.append(g)

                else:
                    a=soup.find_all('h1',attrs = {'class':'raleway title-detail'})[0].get_text()
                    if "asta" in a:
                        next
                    else:
                        b=soup.find_all('div',attrs = {'class':'col-xs-12 description-text text-compressed'})[0].get_text()
                        c=soup.find_all('li' ,attrs={'class':'features__price'})[0].get_text()
                        d=soup.find_all('span',attrs = {'class':'text-bold'})[0].get_text() 
                        e=soup.find_all('span',attrs = {'class':'text-bold'})[1].get_text()
                        f=soup.find_all('span',attrs = {'class':'text-bold'})[2].get_text()
                        g=soup.find_all('abbr',attrs = {'class':'text-bold im-abbr'})[0].get_text()
                        b=b.replace("\n","")
                        c=c.replace("€","")
                        g=g.replace("\n","")
                        g=g.replace("T","0")
                        titles.append(a)
                        descriptions.append(b)
                        prices.append(c)
                        locali.append(d)
                        sup.append(e)
                        bath.append(f)
                        piano.append(g)

            except:
                next
        counturl+=1

    print ("End : %s" % time.ctime())

    
    
def save(): 
    data_tuples = list(zip(titles,descriptions,prices,locali,sup,bath,piano))
    df=pd.DataFrame(data_tuples,columns = ["Title","Description","Prices","Locals","Area","Bath","Floor"])
    
    #clean and save for information clustering
    pd.set_option('display.max_colwidth', -1)
    
    df.to_pickle("housesfinal")
    
    #save description for description clustering
    df = pd.read_pickle("housesfinal")
    des = list(df.Description)
    with open('description.txt', 'w') as f: 
        json.dump(des, f)
        
def clean_information(): 
    
    df = pd.read_pickle("housesfinal")
    # We remove \xa0 from these columns:
    df['Locals'] = df['Locals'].astype(str).str.replace(u'\xa0', '')
    df['Bath'] = df['Bath'].astype(str).str.replace(u'\xa0', '')
    df['Floor'] = df['Floor'].astype(str).str.replace(u'\xa0', '')
    # We remove spaces in these columns:
    df['Prices'] = df['Prices'].astype(str).str.strip()
    df['Floor'] = df['Floor'].astype(str).str.strip()
    # We remove dots in Prices column so, later, we can transform string array to number array:
    df['Prices'] = df['Prices'].str.replace('.', '')
    # We drop rows with non numeric symbols (es. A in Floor column or 3+ in Locals and Bath columns):
    df = df[df.Locals.apply(lambda x: x.isnumeric()) & df.Bath.apply(lambda x: x.isnumeric()) & df.Floor.apply(lambda x: x.isnumeric()) & df.Prices.apply(lambda x: x.isnumeric()) & df.Area.apply(lambda x: x.isnumeric())]
    # index of rows from zero to len(df)
    df.index = np.arange(0,len(df))
    return df

def clean_description(): 
    
    with open('description.txt')as f: 
        data = json.load(f)
    cleaned_data = []
    for i in data: 
        #dadta with stop words removed
        s = []       
        #data with punctuation removed
        p = []
        #stemmed data
        st = []
        
        #removing punctuation 
        tokenizer = RegexpTokenizer(r'\w+')
        p.append(tokenizer.tokenize(i))
        
        #removing stop word         
        stop_words = set(stopwords.words('italian'))                
        s = [w for w in p[0] if not w in stop_words] 
        
        #stem        
        it = nltk.stem.snowball.ItalianStemmer()
        st = [it.stem(j) for j in s]
        
        #add cleaned entry to our list        
        cleaned_data.append(st)
    with open('cleaned_description.txt', 'w') as f: 
        json.dump(cleaned_data, f)    


def create_vocabulary(): 
    
    with open('cleaned_description.txt') as f: 
        c_descriptions = json.load(f)   
        
    words = {w for l in c_descriptions for w in l}
    
    words = list(words)
    
    vocabulary = {words[i]: i for i in range(len(words))}
    
    with open('vocabulary.txt', 'w') as f: 
        json.dump(vocabulary, f)
        
        
        
def tf(): 
    
    with open('cleaned_description.txt')as f: 
        c_d = json.load(f)
    
    with open('vocabulary.txt') as f: 
        voc = json.load(f)
    
    inverted_index = {}
    #inverted_index is like 'word': (doc_id, term_frequency)
    for i in range(len(c_d)): 
        temp = {}
        for w in c_d[i]: 
            if w in temp: 
                temp[w] += 1
            else: 
                temp[w] = 1
                
        l = {voc[k]: (i,temp[k]) for k in temp}
        
        for i in l: 
            if i in inverted_index: 
                inverted_index[i].append(l[i])
            else: 
                inverted_index[i] = [l[i]]            
    
    return inverted_index


def idf(): 
    index = tf()
    with open('description.txt')as f: 
        d = json.load(f)
    tfidf = {}
    for i in index:     
        for v in index[i]: 
            if i in tfidf: 
                tfidf[i].append((v[0], v[1] * math.log(len(d)/len(index[i]))))
            else:
                tfidf[i] = [(v[0], v[1] * math.log(len(d)/len(index[i])))]
    with open('tfidf_index.txt', 'w')as f: 
        json.dump(tfidf, f)  


        
# Using the elbow method to find the optimal number of clusters
def elbow_method(m, v): 
    wcss_m = [] # Within Cluster Sum of Squares
    if(v == 'i'): 
        for k in range(1, 21):
            kmeans_m = KMeans(n_clusters = k, random_state = 0).fit(m)
            wcss_m.append(kmeans_m.inertia_)
    else: 
        for k in range(1, 21):
            kmeans_mw = KMeans(n_clusters = k, init = 'k-means++', random_state = 0).fit(m)
            wcss_m.append(kmeans_mw.inertia_)
    return wcss_m



def index_to_dataframe():
    # Convert the tfi_index to dataframe
    with open('tfidf_index.txt') as f: 
        index = json.load(f)

    tfi_df = pd.DataFrame() 

    for key in index:     
        d = []
        idx = []
        for i in index[key]: 
            d.append(i[1])
            idx.append(i[0])
        s = pd.Series(data = d, index = idx, name = key)
        tfi_df = pd.concat([tfi_df, s], axis=1)
    return tfi_df

        
        
def get_dictionaries(labels1, labels2): 
    # dictionary of labels1: d1

    d1 = {'0':[], '1':[], '2':[], '3':[]}

    for i in range(len(labels1)):
        for j in d1.keys():
            if int(j) == labels1[i]:
                d1[j].append(i)

    # dictionary of labels2: d2

    d2 = {'0':[], '1':[], '2':[], '3':[]}

    for i in range(len(labels2)):
        for j in d2.keys():
            if int(j) == labels2[i]:
                d2[j].append(i)
    return d1,d2



def jaccard(x, y):
    s1 = set(x) # with set we remove all duplicates
    s2 = set(y)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def get_word_clouds(dfthree): 
    
    with open('description.txt')as f: 
        dfd = json.load(f)

    one = dfthree.Announcements[0]
    w1 = []
    for i in one:
        for j in range(len(dfd)):
            if int(i) == j:
                w1.append(dfd[j])

    wc1 = []
    for i in w1:
        a = i.strip()
        b = ''.join([j for j in a if not j.isdigit()])
        c = "".join(k for k in b if k not in string.punctuation)
        d = c.replace("€", "")
        wc1.append(d)
    
    #2nd similar couple of clusters
    two = dfthree.Announcements[1]

    w2 = []
    for i in two:
        for j in range(len(dfd)):
            if int(i) == j:
                w2.append(dfd[j])

    wc2 = []
    for i in w2:
        a = i.strip()
        b = ''.join([j for j in a if not j.isdigit()])
        c = "".join(k for k in b if k not in string.punctuation)
        d = c.replace("€", "")
        wc2.append(d)
        
    #2nd similar couple of clusters
        
    three = dfthree.Announcements[2]

    w3= []
    for i in three:
        for j in range(len(dfd)):
            if int(i) == j:
                w3.append(dfd[j])

    wc3 = []
    for i in w3:
        a = i.strip()
        b = ''.join([j for j in a if not j.isdigit()])
        c = "".join(k for k in b if k not in string.punctuation)
        d = c.replace("€", "")
        wc3.append(d)
        
    return wc1,wc2,wc3

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val
    
#  This function returns a prime number that is close to three times the length of our file(approx. 3*110,000,001). We will need this for hashing(with or without order)  
def get_nearest_prime(old_number):
    nearest_prime = 0    
    for num in range(old_number, 2 * old_number) :        
        bol = False 
        for i in range(2,num):                  
            if num % i == 0:                
                bol = True
                break
        if(bol == False):
            near_to_two = False
            for j in range(num-2, num+2): 
                if np.log2(j)%2 == 0: 
                    near_to_two = True
                    break
            if(near_to_two == False): 
                nearest_prime = num
                break
    return nearest_prime


def hash_without_order(a, i):    
    hash_table = defaultdict(list)  
    '''
    M is a prime number. We are going to module our large number(converted from the string) by a prime number. 
    Because prime numbers have less factors so they are less likely to give the same value. 
    In order to avoid collison when hashing we should assign more storage space than needed. 
    We are going to reserve 3 times more than what is needed. 
    Then we are going to find the nearest prime number to this value(3*length(file)).     
    '''
    #nearest prime number of 3 time more than the length of our file not close to power of n of 
    M = 330000023
    
    #I used this to try for 1000 rows
    #M = 3001
    
    for s in range(len(a)): 
        ords = [ord(i)**2 for i in a[s]] 
        
        h = ((sum(ords)*104729)%M)         
        hash_table[h].append(sum(ords)*104729)
        
    try: 
        with open('withoutOrder/withoutOrder' + str(i) +'.txt', 'a') as f:
            json_str = json.dump(hash_table, f)
    except: 
        os.mkdir('withoutOrder')
        with open('withoutOrder/withoutOrder' + str(i) +'.txt', 'w') as f:
            json_str = json.dump(hash_table, f)
            
            
def hash_with_order(a,j):
    hash_table = defaultdict(list)
    #nearest prime number of 3 time more than the length of our file not close to power of n of 
    M = 330000023
    
    #I used this to try for 1000 rows
    #M = 3001 
    
    for s in range(len(a)): 
        sum = 0
        for i in range(len(a[s])):         
            '''
            Unlike hasing with out order, we care about order here. So the value of a character must depend on its position. 
            To do that we get the unicode of a character and square it. If the character isn't the first character of the string
            we multiple the squared unicode value by the previous sum divided by the index of the character. 
            '''
            if(sum != 0):
                sum += ((ord(a[s][i])**2) * (sum/(i+1))) 
            else :
                sum += ((ord(a[s][i])**2))                   
        
        h = ((sum*104729)%M)         
        hash_table[h].append(sum*104729)
        
    try: 
        with open('withOrder/withOrder' + str(j) +'.txt', 'a') as f:
            json_str = json.dump(hash_table, f)
    except: 
        os.mkdir('withOrder')
        with open('withOrder/withOrder' + str(j) +'.txt', 'w') as f:
            json_str = json.dump(hash_table, f)

            
            
# We just use this function to load hashing tables(with or without order, it depend of order value)                
def get_file(order):
    global a
    global idx
    
    idx = 0
    error = 1
    a = defaultdict(list)
    
    if(order == 1):         
        while(True):            
            try: 
                with open('withOrder/withOrder'+ str(idx) + '.txt') as f: 
                    file = json.load(f)  
                for h in file:
                    if(h != "NaN"):
                        for v in file[h]:                                                       
                            a[h].append(v)
            except:
                error += 1
                if(error > 2): 
                    break            
                else: 
                    idx += 1
                    continue          
            idx += 1
    else: 
        while(True):            
            try: 
                with open('withoutOrder/withoutOrder'+ str(idx) + '.txt') as f:                 
                    file = json.load(f)            
                for h in file:
                    if(h != "NaN"):
                        for v in file[h]:                                                       
                            a[h].append(v)
            except:
                error += 1
                if(error > 2): 
                    break            
                else: 
                    idx += 1
                    continue
            idx += 1             
        

#For the dictionary we have created, we check for every keys if we have multiple values. If we do, we say that they may be duplicates.
def check_duplicates(order):
    
    get_file(order)    
    file = a
    duplicates = {i: file[i] for i in file if len(file[i]) > 1 }        
    print(str(len(duplicates)) + ' word/s was/were duplicated out of ' + str((idx+1)*(1000)) +' rows.')
    
    return duplicates    


#We just transform our list into a set and check for the length for every key. If the value of the length is > than 1 we have false positives(because we are assuming that our function is working properly).
def check_duplicates_falsePositive(order):   
    
    duplicates = check_duplicates(order)    
    falsePositive = 0 
    
    for i in duplicates: 
        temp = set()
        for v in duplicates[i]:            
            temp.add(v)
        if len(temp) > 1: 
            falsePositive += 1
            
    print('There were ' + str(falsePositive) + ' false positives.')    
    

