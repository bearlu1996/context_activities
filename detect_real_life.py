
n_context = 3
threshold = 0.65

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import random
from numpy import insert
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer

log = xes_importer.apply('sepsis.xes.gz')




dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)


dataframe = dataframe[["case:concept:name", "concept:name"]]

print(dataframe)

print("_________________________________________________________________")

context_data = []
current_row_id = 0
all_activities = set()
for row in dataframe.iterrows():
        current_trace = dataframe.iat[current_row_id, 0]
        current_activity = dataframe.iat[current_row_id, 1]
        tmp_data = []
        for i in range(0, n_context):
            a = 3 - i
            if current_row_id - a >= 0 and dataframe.iat[current_row_id - a, 0] == current_trace:
                tmp_data.append(dataframe.iat[current_row_id - a, 1])
            else:
                tmp_data.append("0")
        for i in range(1, n_context + 1):
            if current_row_id + i <= dataframe.shape[0] - 1 and dataframe.iat[current_row_id + i, 0] == current_trace:
                tmp_data.append(dataframe.iat[current_row_id + i, 1])
            else:
                tmp_data.append("0")    
        current_row_id = current_row_id + 1
        tmp_data.append(current_activity)
        all_activities.add(current_activity)
        tmp_data.append(current_row_id + 1)
        context_data.append(tmp_data)

column_list = []
for i in range(0, n_context):
    column_list.append("event_Prev" + str(i + 1))
for i in range(0, n_context):
    column_list.append("event_Back" + str(i + 1))
column_list.append("event")
column_list.append("id")
context_dataframe = pd.DataFrame(context_data, columns = column_list)
print(context_dataframe)



import gensim
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import pandas as pd
from gensim.models.doc2vec import TaggedDocument


def learn(folderName,vectorsize,data):
   
    documents = data[["event_Prev1","event_Prev2","event_Prev3", "event_Back1","event_Back2", "event_Back3"]].to_numpy()
    index = 0
    taggeddoc = []
    for a in documents:
        wordslist = []
        for b in a:
            wordslist.append(str(b))
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(wordslist))).split(),[index])
        taggeddoc.append(td)
        index= index +1
    
    print ('Data Loading finished, ', str(len(taggeddoc)), ' traces found.')
    model = gensim.models.Doc2Vec(taggeddoc, dm = 0, alpha=0.025, vector_size= vectorsize, window=3, min_alpha=0.025, min_count=0)

# start training
    nrEpochs= 10
    for epoch in range(nrEpochs):
        if epoch % 2 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(taggeddoc,total_examples=len(taggeddoc), epochs=nrEpochs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay


    model.save('output/'+folderName+'T2VVS'+str(vectorsize) +'.model')
    model.save_word2vec_format('output/'+folderName+ 'T2VVS'+str(vectorsize) + '.word2vec')


def cluster(folderName, vectorsize, data):
  
    documents = data[["event_Prev1","event_Prev2","event_Prev3", "event_Back1","event_Back2", "event_Back3"]].to_numpy()
    events_df = data["event"]
    
    index = 0
    corpus = []
    for a in documents:
        wordslist = []
        for b in a:
            wordslist.append(str(b))
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(wordslist))).split(),[index])
        corpus.append(td)
        index= index +1

    print ('Data Loading finished, ', str(len(corpus)), ' traces found.')


    model= gensim.models.Doc2Vec.load('output/'+folderName+'T2VVS'+str(vectorsize) +'.model')

    vectors = []
    events = []
    NUM_CLUSTERS= 3
    print("inferring vectors")

    for current in events_df:
        events.append(str(current))
    
    for doc_id in range(len(corpus)):
        #print(corpus[doc_id].words)
        model.random.seed(0)
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        vectors.append(inferred_vector)
        #events.append(corpus[doc_id].words[5])
    print("done")
    
    db_a = DBSCAN(eps=0.1, min_samples=5).fit(vectors)
    assigned_clusters = db_a.labels_
    print(len(set(assigned_clusters)))
    print("cluster done")
    
    #trace_list = loadXES.get_trace_names(folderName+".xes")
    trace_list = data["id"].to_numpy().tolist()
    #print(trace_list)
    clusterResult= {}
    for doc_id in range(len(corpus)):
        clusterResult[trace_list[doc_id]]=assigned_clusters[doc_id]

    filter = []
    events_filter = []
    import scipy.spatial as ss
    import numpy as np
    print(type(vectors))
    events_filter_clusters = {}
    for i in range(len(vectors)):
        #print(vectors[i])
        if assigned_clusters[i] != -1:
            filter.append(vectors[i])
            events_filter.append(events[i])
            if assigned_clusters[i] not in events_filter_clusters:
                events_filter_clusters[assigned_clusters[i]] = []
            events_filter_clusters[assigned_clusters[i]].append(vectors[i])
    total = 0
    import numpy
    for i in events_filter_clusters:
        test = numpy.array(events_filter_clusters[i])
        #big_convex_hull_test = test
        try:
           
            
            hull = ss.ConvexHull(test)
          

            total = total + hull.volume
        except:
            continue
        print(total)
    
    #print(test)
    
    
    print('Total: ',total)
    context_activities = []
    for i in all_activities:
      
        points = []
        for a in range(len(filter)):
            if events_filter[a] == i:
                points.append(filter[a])
        b = DBSCAN(eps=0.1, min_samples=5).fit(points)
        assigned_clusters = b.labels_
        print(len(set(assigned_clusters)))

        points_filter = []
        events_filter_clusters = {}

        for a in range(len(points)):
            if assigned_clusters[a] != -1:
                points_filter.append(points[a])
                if assigned_clusters[a] not in events_filter_clusters:
                    events_filter_clusters[assigned_clusters[a]] = []
                events_filter_clusters[assigned_clusters[a]].append(points[a])
        test = numpy.array(points_filter)
        
        
        total_tmp = 0
        print(len(events_filter_clusters))
        for a in events_filter_clusters:
            test = numpy.array(events_filter_clusters[a])
            try:
                hull = ss.ConvexHull(test)
               
                total_tmp = total_tmp + hull.volume
            except Exception as e:
                continue
           
        print(str(i) + ': ' + str(total_tmp))
        print(str(i) + ': ' + str(total_tmp / total))    
        if (total_tmp / total) > threshold:
            context_activities.append(i)
    
    for current_activity in context_activities:
        print(current_activity)
        


vectorsize=3
learn('test',vectorsize,context_dataframe)
cluster('test',vectorsize,context_dataframe)
