#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


# Read the data from amazon-books.csv into amazonBooks dataframe;
amazonBooks = pd.read_csv('./amazon-books.csv', index_col=0)


# In[3]:


amazonBooks.head()


# In[4]:


amazonBooks.loc[["0805047905"],["Categories","Title"]]


# In[5]:


# Read the data from amazon-books-copurchase.adjlist;
# assign it to copurchaseGraph weighted Graph;
# node = ASIN, edge= copurchase, edge weight = category similarity
fhr=open("amazon-books-copurchase.edgelist", 'rb')
copurchaseGraph=nx.read_weighted_edgelist(fhr)
fhr.close()


# In[6]:


# Now let's assume a person is considering buying the following book;
# what else can we recommend to them based on copurchase behavior 
# we've seen from other users?
print ("Looking for Recommendations for Customer Purchasing this Book:")
print ("--------------------------------------------------------------")
purchasedAsin = '0805047905'

# Let's first get some metadata associated with this book
print ("ASIN = ", purchasedAsin) 
print ("Title = ", amazonBooks.loc[purchasedAsin,'Title'])
print ("SalesRank = ", amazonBooks.loc[purchasedAsin,'SalesRank'])
print ("TotalReviews = ", amazonBooks.loc[purchasedAsin,'TotalReviews'])
print ("AvgRating = ", amazonBooks.loc[purchasedAsin,'AvgRating'])
print ("DegreeCentrality = ", amazonBooks.loc[purchasedAsin,'DegreeCentrality'])
print ("ClusteringCoeff = ", amazonBooks.loc[purchasedAsin,'ClusteringCoeff'])


# In[7]:


# Now let's look at the ego network associated with purchasedAsin in the
# copurchaseGraph - which is esentially comprised of all the books 
# that have been copurchased with this book in the past
# (1) YOUR CODE HERE: 
#     Get the depth-1 ego network of purchasedAsin from copurchaseGraph,
#     and assign the resulting graph to purchasedAsinEgoGraph

purchasedAsinEgoGraph = nx.ego_graph(copurchaseGraph, purchasedAsin, radius=1)
print (purchasedAsinEgoGraph.neighbors(purchasedAsin))


# In[8]:


nx.draw(purchasedAsinEgoGraph)
plt.show()


# In[9]:


# Next, recall that the edge weights in the copurchaseGraph is a measure of
# the similarity between the books connected by the edge. So we can use the 
# island method to only retain those books that are highly simialr to the 
# purchasedAsin
# (2) YOUR CODE HERE: 
#     Use the island method on purchasedAsinEgoGraph to only retain edges with 
#     threshold >= 0.5, and assign resulting graph to purchasedAsinEgoTrimGraph
threshold = 0.5
purchasedAsinEgoTrimGraph = nx.Graph()
for f, t, e, in purchasedAsinEgoGraph.edges(data=True):
    if e["weight"]>= threshold:
        purchasedAsinEgoTrimGraph.add_edge(f,t,weight=e['weight'])
        


# In[10]:


nx.draw(purchasedAsinEgoTrimGraph)
plt.show()


# In[11]:


# Next, recall that given the purchasedAsinEgoTrimGraph you constructed above, 
# you can get at the list of nodes connected to the purchasedAsin by a single 
# hop (called the neighbors of the purchasedAsin) 
# (3) YOUR CODE HERE: 
#     Find the list of neighbors of the purchasedAsin in the 
#     purchasedAsinEgoTrimGraph, and assign it to purchasedAsinNeighbors
purchasedAsinNeighbors = []
for f, t, e in purchasedAsinEgoTrimGraph.edges(data=True):
    if f == purchasedAsin:
        purchasedAsinNeighbors.append(t)
print(purchasedAsinNeighbors)


# In[12]:


len(purchasedAsinNeighbors)


# In[13]:


#check for missing data in main dataframe
amazonBooks.isna().sum()


# In[14]:


#Impute "MISSING" for missing data cells
from sklearn.impute import SimpleImputer
si = SimpleImputer(missing_values=np.nan, strategy="constant",fill_value="MISSING")
amazonBooks["ImpCategories"] = pd.DataFrame(si.fit_transform(amazonBooks[["Categories"]]),index=amazonBooks.index)
#amazonBooks.head()


# In[15]:


#Recheck for missing data 
amazonBooks = amazonBooks.drop(["Categories"],axis=1)
amazonBooks.isna().sum()


# In[16]:


amazonBooks.head()


# In[17]:


# Next, let's pick the Top Five book recommendations from among the 
# purchasedAsinNeighbors based on one or more of the following data of the 
# neighboring nodes: SalesRank, AvgRating, TotalReviews, DegreeCentrality, 
# and ClusteringCoeff
# (4) YOUR CODE HERE: 
#     Note that, given an asin, you can get at the metadata associated with  
#     it using amazonBooks (similar to lines 29-36 above).
#     Now, come up with a composite measure to make Top Five book 
#     recommendations based on one or more of the following metrics associated 
#     with nodes in purchasedAsinNeighbors: SalesRank, AvgRating, 
#     TotalReviews, DegreeCentrality, and ClusteringCoeff. Feel free to compute
#     and include other measures if you like.
#     YOU MUST come up with a composite measure.
#     DO NOT simply make recommendations based on sorting!!!
#     Also, remember to transform the data appropriately using 
#     sklearn preprocessing so the composite measure isn't overwhelmed 
#     by measures which are on a higher scale.


# In[18]:


#Calculate node measures for each neighborhood nodes

n = purchasedAsin

# get the clustering coefficient of the ego node 
cc = nx.clustering(purchasedAsinEgoTrimGraph, n)
print ("Clustering Coef for", n,"is:", round(cc,2))

# get the average clustering coefficient of the ego network
acc = nx.average_clustering(purchasedAsinEgoTrimGraph)
print ("Avg Clustering Coef for", n,"is:", round(acc,2))


# In[19]:


#Clustering coefficient dictionary
#Calculate CC of each node in purchasedAsinEgoTrimGraph
clustCoeff_dict = {}
for n in nx.nodes(purchasedAsinEgoTrimGraph):
    cc = nx.clustering(purchasedAsinEgoTrimGraph, n)
    items = {n:cc}
    clustCoeff_dict.update(items)
print(clustCoeff_dict)
    


# In[20]:


#Calculate degree centrality measure for each node in the neighborhood
degreeCentrality = nx.degree(purchasedAsinEgoTrimGraph)
#print(degreeCentrality)


# In[21]:


#Degree Centrality Dictionary
#Create a dictonary where keys=nodes in neighborhood and values=degree centrality measure
degreeCentrality_dict = {}
for a,b in degreeCentrality:
    items = {a:b}
    degreeCentrality_dict.update(items)
print(degreeCentrality_dict)


# In[22]:


#Total reviews dictionary
totalReviews_dict = {}
for index, row in amazonBooks.iterrows():
    n = index
    #print(n)
    if n in list(purchasedAsinNeighbors):
        reviews = (row["TotalReviews"])
        #print(n,reviews)
        new = {n:reviews}
        totalReviews_dict.update(new)

print(totalReviews_dict)
        


# In[23]:


avgRating_dict = {}
for index, row in amazonBooks.iterrows():
    n = index
    #print(n)
    if n in list(purchasedAsinNeighbors):
        rating = (row["AvgRating"])
        #print(n,reviews)
        new = {n:rating}
        avgRating_dict.update(new)

print(avgRating_dict)
 


# In[24]:


#Composite score for each node
import math
composite = {}

for a,d in degreeCentrality_dict.items():
    for b,c in clustCoeff_dict.items():
        for q,r in avgRating_dict.items():
            for s,t in totalReviews_dict.items():
                if a == b and (b == q and q == s):
                    x = (math.log(t+1)*0.3) * d + ((3*r) * c)

                    items = {a:x}
                    composite.update(items)
print(composite)
    


# In[25]:


#sorting the composite score
composite_sorted = {k: v for k, v in sorted(composite.items(), key=lambda item: item[1], reverse = True)}
print(composite_sorted)


# In[26]:


#Fiding top 5 from the sorted compostite score
sorted_top5 = {}
c=1
for i in composite_sorted:
    if c > 5:
        break
    sorted_top5[i] = composite_sorted[i]
    c+=1
print(sorted_top5)


# Print Top 5 recommendations (ASIN, and associated Title, Sales Rank, 
# TotalReviews, AvgRating, DegreeCentrality, ClusteringCoeff)
# (5) YOUR CODE HERE:
n = 1
for i in sorted_top5:
    print("Top ",n," recommendation for the Book with Asin '0805047905' is: ")
    print("Asin = ", i)
    #print("Title of the Book = ", amazonBooks[i]['Title'])
    #print("SalesRank = ", amazonBooks[i]['SalesRank'])
    #print("TotalReviews = ", amazonBooks[i]['TotalReviews'])
    #print("AvgRating = ", amazonBooks[i]['AvgRating'])
    #print("DegreeCentrality = ", amazonBooks[i]['DegreeCentrality'])
    #print("ClusteringCoeff = ", amazonBooks[i]['ClusteringCoeff'])
    #print()
    #n += 1


# In[ ]:




