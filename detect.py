num_of_inserted_activities = 1
percentage_of_inserted_activities = 0.15
percentage_of_noises = 0
n_context = 3


from unittest import result
import pandas as pd
def run(input_log_name):
    #read and convert event log into dataframe
    import random
    from numpy import insert
    from pm4py.objects.log.importer.xes import importer as xes_importer
    import pandas as pd
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer

    #log = xes_importer.apply('BPM2016\imprInLoop_adaptive_OD\mrt06-1911\logs\AM_1_LogR_ILP_Sequence_mrt06-1911.xes.gz')
    log = xes_importer.apply(input_log_name)
    #net, initial_marking, final_marking = inductive_miner.apply(log)
    #gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    #pn_visualizer.view(gviz)



    dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

    to_insert = int(dataframe.shape[0] * percentage_of_inserted_activities)
    to_insert_noise = int(dataframe.shape[0] * percentage_of_noises)
    dataframe = dataframe[["case:concept:name", "concept:name"]]
    #randomly insert context events
    for i in range(0, num_of_inserted_activities):
        dataframe = dataframe[["case:concept:name", "concept:name"]]
        ##print(dataframe.iat[1, 1])
        ##print(dataframe.shape[0])
        #break
        for a in range(0, to_insert):
            insert_index = random.randint(1, dataframe.shape[0] - 1)
            insert_case_id = dataframe.iat[insert_index, 0]
            dataframe.loc[insert_index + 0.5] = insert_case_id, "insert_" + str(i)
            dataframe = dataframe.sort_index().reset_index(drop=True)
        
    for i in range(0, num_of_inserted_activities):
        dataframe = dataframe[["case:concept:name", "concept:name"]]
        ##print(dataframe.iat[1, 1])
        ##print(dataframe.shape[0])
        #break
        for a in range(0, to_insert_noise):
            insert_index = random.randint(1, dataframe.shape[0] - 1)
            insert_case_id = dataframe.iat[insert_index, 0]
            dataframe.loc[insert_index + 0.5] = insert_case_id, "noise_" + str(i)
            #dataframe.loc[insert_index + 0.5] = insert_case_id, "A"
            dataframe = dataframe.sort_index().reset_index(drop=True)

    #print("Events have been inserted.")
    #print("_________________________________________________________________")


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
    #print(context_dataframe)



    import gensim
    import nltk
    from nltk.cluster.kmeans import KMeansClusterer
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    import pandas as pd
    from gensim.models.doc2vec import TaggedDocument


    def learn(folderName,vectorsize,data):
        #documents = loadXES.get_doc_XES_tagged(folderName+'.xes')

        ##print(data)
        #data = pd.read_csv("inserted_3.csv_train.csv")
        #documents = data[["event_Prev5","event_Prev6","event_Prev7","event_Prev8","event_Prev9", "event_Back9","event_Back8","event_Back7","event_Back6","event_Back5"]].to_numpy()
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

        #print ('Data Loading finished, ', str(len(taggeddoc)), ' traces found.')
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
        #corpus = loadXES.get_doc_XES_tagged(folderName+'.xes')

        #data = pd.read_csv("inserted_3.csv_train.csv")
        #data = data.loc[data['event'] == 4]
        #data = data.head(10)
        ##print(data)
        #documents = data[["event_Prev5","event_Prev6","event_Prev7","event_Prev8","event_Prev9", "event_Back9","event_Back8","event_Back7","event_Back6","event_Back5"]].to_numpy()
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

        #print ('Data Loading finished, ', str(len(corpus)), ' traces found.')


        model= gensim.models.Doc2Vec.load('output/'+folderName+'T2VVS'+str(vectorsize) +'.model')

        vectors = []
        events = []
        NUM_CLUSTERS= 3
        #print("inferring vectors")

        for current in events_df:
            events.append(str(current))

        for doc_id in range(len(corpus)):
            ##print(corpus[doc_id].words)
            model.random.seed(0)
            inferred_vector = model.infer_vector(corpus[doc_id].words)
            vectors.append(inferred_vector)
            #events.append(corpus[doc_id].words[5])
        #print("done")

        db_a = DBSCAN(eps=0.5, min_samples=5).fit(vectors)
        assigned_clusters = db_a.labels_
        #print(len(set(assigned_clusters)))
        #print("cluster done")

        #trace_list = loadXES.get_trace_names(folderName+".xes")
        trace_list = data["id"].to_numpy().tolist()
        ##print(trace_list)
        clusterResult= {}
        for doc_id in range(len(corpus)):
            clusterResult[trace_list[doc_id]]=assigned_clusters[doc_id]

        filter = []
        events_filter = []
        import scipy.spatial as ss
        import numpy as np
        #print(type(vectors))
        events_filter_clusters = {}
        for i in range(len(vectors)):
            ##print(vectors[i])
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
            try:
                hull = ss.ConvexHull(test)
                total = total + hull.volume
            except:
                continue
            #print(total)

        ##print(test)


        #print('Total: ',total)
        context_activities_5 = []
        context_activities_55 = []
        context_activities_6 = []
        context_activities_65 = []
        context_activities_7 = []
        context_activities_75 = []
        context_activities_8 = []
        context_activities_85 = []
        context_activities_9 = []
        context_activities_95 = []
        context_activities_final_dic = {}
        context_activities_final = []
        for i in all_activities:
            points = []
            for a in range(len(filter)):
                if events_filter[a] == i:
                    points.append(filter[a])
            if(len(points) == 0):
                continue
            b = DBSCAN(eps=0.1, min_samples=5).fit(points)
            assigned_clusters = b.labels_
            #print(len(set(assigned_clusters)))

            points_filter = []
            events_filter_clusters = {}

            for a in range(len(points)):
                if assigned_clusters[a] != -1:
                    points_filter.append(points[a])
                    if assigned_clusters[a] not in events_filter_clusters:
                        events_filter_clusters[assigned_clusters[a]] = []
                    events_filter_clusters[assigned_clusters[a]].append(points[a])
            test = numpy.array(points_filter)
            ##print(len(test[0]))

            total_tmp = 0
            for a in events_filter_clusters:
                test = numpy.array(events_filter_clusters[a])
                try:
                    hull = ss.ConvexHull(test)
                    total_tmp = total_tmp + hull.volume
                except:
                    continue
                ##print(total_tmp)
            #print(str(i) + ': ' + str(total_tmp))
            #print(str(i) + ': ' + str(total_tmp / total))
            context_activities_final_dic[i] = total_tmp / total
            if (total_tmp / total) >= 0.5:
                context_activities_5.append(i)
            if (total_tmp / total) >= 0.55:
                context_activities_55.append(i)
            if (total_tmp / total) >= 0.6:
                context_activities_6.append(i)
            if (total_tmp / total) >= 0.65:
                context_activities_65.append(i)
            if (total_tmp / total) >= 0.7:
                context_activities_7.append(i)
            if (total_tmp / total) >= 0.75:
                context_activities_75.append(i)
            if (total_tmp / total) >= 0.8:
                context_activities_8.append(i)
            if (total_tmp / total) >= 0.85:
                context_activities_85.append(i)
            if (total_tmp / total) >= 0.9:
                context_activities_9.append(i)
            if (total_tmp / total) >= 0.95:
                context_activities_95.append(i)

        from operator import itemgetter  
        for key, value in sorted(context_activities_final_dic.items(), key = itemgetter(1), reverse = True):
            context_activities_final.append(key)
            if len(context_activities_final) == num_of_inserted_activities:
                break

        tp = 0
        fp = 0
        fn = 0
        new_row = []
        new_row.append(input_log_name)
        identified_activities = ""
        for current_activity in context_activities_5:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)

        tp = 0
        fp = 0
        fn = 0
       
        identified_activities = ""
        for current_activity in context_activities_55:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)

        tp = 0
        fp = 0
        fn = 0
        
        identified_activities = ""
        for current_activity in context_activities_6:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)

        tp = 0
        fp = 0
        fn = 0
        
        identified_activities = ""
        for current_activity in context_activities_65:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)

        tp = 0
        fp = 0
        fn = 0
        
        identified_activities = ""
        for current_activity in context_activities_7:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)

        tp = 0
        fp = 0
        fn = 0
        
        identified_activities = ""
        for current_activity in context_activities_75:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)

        tp = 0
        fp = 0
        fn = 0

        tp = 0
        fp = 0
        fn = 0
        
        identified_activities = ""
        for current_activity in context_activities_8:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)

        tp = 0
        fp = 0
        fn = 0
        
        identified_activities = ""
        for current_activity in context_activities_85:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)

        tp = 0
        fp = 0
        fn = 0
        
        identified_activities = ""
        for current_activity in context_activities_9:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)

        tp = 0
        fp = 0
        fn = 0
        
        identified_activities = ""
        for current_activity in context_activities_95:
            ##print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)
        
        tp = 0
        fp = 0
        fn = 0

        identified_activities = ""
        for current_activity in context_activities_final:
            print(current_activity)
            identified_activities = identified_activities + " | " + current_activity
            if "insert" in current_activity:
                tp = tp + 1
            else:
                fp = fp + 1
        fn = num_of_inserted_activities - tp
        ##print("tp: " + str(tp))
        ##print("fp: " + str(fp))
        ##print("fn: " + str(fn))
        new_row.append(tp)
        new_row.append(fp)
        new_row.append(fn)
        new_row.append(identified_activities)
        
        results.append(new_row)
    logName='test'
    vectorsize=3
    learn(logName,vectorsize,context_dataframe)
    cluster(logName,vectorsize,context_dataframe)

#run()
results = []
folders = []

#run("BPM2016/imprInLoop_adaptive_OD/mrt06-1652/logs/CG_1_LogR_IM_Sequence_mrt06-1652.xes.gz")
#final_frame = pd.DataFrame(results, columns = ["log", "tp_5", "fp_5", "fn_5", "results_5", "tp_55", "fp_55", "fn_55", "results_55", "tp_6", "fp_6", "fn_6", "results_6", "tp_65", "fp_65", "fn_65", "results_65", "tp_7", "fp_7", "fn_7", "results_7", "tp_75", "fp_75", "fn_75", "results_75", "tp_final", "fp_final", "fn_final", "results_final"])
##print(final_frame)
#final_frame.to_csv(str(num_of_inserted_activities) + "_inserted_activities_" + str(percentage_of_inserted_activities) + "_" + str(n_context) + "_result.csv", encoding='utf-8', index=False)



import glob
files = glob.glob("BPM2016/*")
for file in files:
    if "." not in file:
        #if "imprInLoop_adaptive_OD" in file:
        #    continue
        files_2 = glob.glob("BPM2016/" + file.split("\\")[1] + "/*")
        for file_2 in files_2:
           files_3 = glob.glob("BPM2016/" + file.split("\\")[1] + "/" + file_2.split("\\")[1] + "/logs/*.xes.gz")
           for file_3 in files_3:
               if "LogD" in file_3 or "LogR" in file_3:
                   continue
               print("BPM2016/" + file.split("\\")[1] + "/" + file_2.split("\\")[1] + "/logs/" + file_3.split("\\")[1])
               #for count in range(0, 5):
               try:
                   run("BPM2016/" + file.split("\\")[1] + "/" + file_2.split("\\")[1] + "/logs/" + file_3.split("\\")[1])
               except:
                   #print("exception")
                   continue
        final_frame = pd.DataFrame(results, columns = ["log", "tp_5", "fp_5", "fn_5", "results_5", "tp_55", "fp_55", "fn_55", "results_55", "tp_6", "fp_6", "fn_6", "results_6", "tp_65", "fp_65", "fn_65", "results_65", "tp_7", "fp_7", "fn_7", "results_7", "tp_75", "fp_75", "fn_75", "results_75", "tp_8", "fp_8", "fn_8", "results_8", "tp_85", "fp_85", "fn_85", "results_85", "tp_9", "fp_9", "fn_9", "results_9", "tp_95", "fp_95", "fn_95", "results_95", "tp_final", "fp_final", "fn_final", "results_final"])
#print(final_frame)
#final_frame.to_csv(num_of_inserted_activities + "_inserted_activities_" + percentage_of_inserted_activities + "_" + n_context + "_" + threshold + "_result.csv", encoding='utf-8', index=False)
        final_frame.to_csv(str(num_of_inserted_activities) + "_inserted_activities_" + str(percentage_of_inserted_activities) + "_" + str(n_context) + "_" + file.split("\\")[1] + "_" + str(percentage_of_noises) + "_no_result.csv", encoding='utf-8', index=False)
        results = []