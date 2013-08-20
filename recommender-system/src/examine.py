from collections import defaultdict
from datetime import timedelta,datetime
import json

import numpy

from data.kasteren import load_scikit as load_kasteren
from data.mavlab import load_scikit as load_mavlab
from classifiers.randomc import RandomClassifier
from classifiers.bayes import NaiveBayesClassifier
from classifiers.temporal import TemporalEvidencesClassifier
from classifiers.metrics import multiple_predictions_scores,num_predictions
from classifiers.binners import StaticBinning
from classifiers.metrics import confusion_matrix as calculate_confusion_matrix
from classifiers.postprocess import dynamic_cutoff
from classifiers.caching import PreviousItemCache
import plot

def plot_evidences(dataset):
   cls = TemporalEvidencesClassifier(dataset.features,dataset.target_names,binning_method=StaticBinning(bins=list(range(0,300,10))))
   #cls=NaiveBayesClassifier(dataset.features,dataset.target_names)
   cls = cls.fit(dataset.data, dataset.target)
   plot.plot_evidences(cls)
   #results=cls.predict(dataset.data)

def write_evidences(dataset):
   if not dataset.name=="houseA":
      raise Exception("Wrong dataset")
   cls = TemporalEvidencesClassifier(dataset.features,dataset.target_names,binning_method=StaticBinning(bins=list(range(0,310,10))))
   cls = cls.fit(dataset.data, dataset.target)
   bins=cls.binning_method.bins
   source = cls.sources[("Frontdoor","Closed")]
   interesting_actions={"Frontdoor=Open":"Open front door","Fridge=Open":"Open fridge",
                        "Hall-Bathroom_door=Open":"Open bathroom door","Cups_cupboard=Closed":"Open cups cabinet"}
   observations=dict()
   for a,action in enumerate(cls.target_names):
       if action in interesting_actions:
          observations[interesting_actions[action]]=[source.temporal_counts[b][a] for b in range(len(bins))]
   print json.dumps({"bins":bins,"observations":observations},sort_keys=True,indent=4, separators=(',', ': '))

def timeline(dataset):

    cache=PreviousItemCache()
    cls = TemporalEvidencesClassifier(dataset.features,dataset.target_names,cache=cache)
    cls = cls.fit(dataset.data, dataset.target)

    cache_hit_runtimes=[]
    cache_miss_runtimes=[]
    
    for i,data in enumerate(dataset.timeline(timedelta(seconds=1))):
          runtime=datetime.now()
          cls.predict([data],include_conflict_theta=True)
          runtime=(datetime.now()-runtime).microseconds
          
          if cache.get_statistics()[0]>len(cache_hit_runtimes):
             cache_hit_runtimes.append(runtime)
          else:
             cache_miss_runtimes.append(runtime)  

          #if i>35000: 
          #   break 
          
    print "hits %d, misses %d"%cache.get_statistics()
    print "hit runtime %.4f"%numpy.mean(numpy.array(cache_hit_runtimes))
    print "miss runtime %.4f"%numpy.mean(numpy.array(cache_miss_runtimes))
    


def confusion_matrix(dataset):
    cls = NaiveBayesClassifier(dataset.features,dataset.target_names)
    cls = cls.fit(dataset.data, dataset.target)
    results=cls.predict(dataset.data)
    (labels,matrix)=calculate_confusion_matrix(dataset.target,results)

    letters=list(map(chr, list(range(97, 123))))+list(map(chr, list(range(65, 91))))
    column_width=len(str(matrix.max()))+2
    print "".join([letter.rjust(column_width) for letter in letters[0:len(labels)]])+"     predicted/actual"
    for row,label,letter in zip(matrix,labels,letters):
        print "".join([str(c).rjust(column_width) for c in row])+"   "+letter+" "+label

    labels,tps,fps,fns,weights=count_tpfpfn(dataset.target,results,1)
    for l,tp,fp,fn in zip(labels,tps,fps,fns):
        print l,"tp:",tp,"fp:",fp,"fn:",fn
    

def histogram_compare_methods(dataset):
    #clss = [("NaiveBayes",NaiveBayesClassifier(dataset.features,dataset.target_names)),
    #        ("Temporal Evidences",TemporalEvidencesClassifier(dataset.features,dataset.target_names))]

    intervals_to_test=[("up to 60",list(range(0,60,10))), 
                       ("up to 300",list(range(0,60,10))+list(range(60,300,30))),
                       ("up to 600",list(range(0,60,10))+list(range(60,600,30))),
                       ("up to 600, all short",list(range(0,600,10)))]
    clss = [(name,TemporalEvidencesClassifier(dataset.features,dataset.target_names,binning_method=StaticBinning(bins=bins))) for (name,bins) in intervals_to_test]
    cutoff=1
  
    histogram_data=defaultdict(list)
    for (cls_name,cls) in clss:
        cls = cls.fit(dataset.data, dataset.target) 
        results=cls.predict(dataset.data)
        executed_by_service=defaultdict(int)
        correct_by_service=defaultdict(int)
        for (actual,predictions) in zip(dataset.target,results):
            executed_by_service[actual]+=1
            if actual in predictions[0:cutoff]:
               correct_by_service[actual]+=1
        histogram_data[cls_name]=[(service,executed_by_service[service],correct_by_service[service]) for service in sorted(executed_by_service)]
 
    plot.histogram([cls_name for (cls_name,cls) in clss],histogram_data,"../plots/"+dataset.name+"/histogram_compare_methods.pdf")

def histogram_compare_cutoffs(dataset):
    to_compare=[1,2,3,4]
    cls = TemporalEvidencesClassifier(dataset.features,dataset.target_names)
    cls = NaiveBayesClassifier(dataset.features,dataset.target_names)
    cls = cls.fit(dataset.data, dataset.target)
    results=cls.predict(dataset.data)

    results_by_service=defaultdict(list)
    position = lambda predictions,actual : predictions.index(actual)+1 if actual in predictions else None
    for (actual,predictions) in zip(dataset.target,results):
        results_by_service[actual].append(position(predictions,actual))
    
    count_correct = lambda positions,cutoff : len([p for p in positions if p<=cutoff])
    histogram_data=dict()
    for c in to_compare:
        histogram_data[c]=[(service,len(service_results),count_correct(service_results,c)) for (service,service_results) in sorted(results_by_service.items())]
    plot.histogram(to_compare,histogram_data,"../plots/"+dataset.name+"/histogram_compare_cutoffs.pdf")
    

def scatter_conflict_theta(dataset):

    #aggregate scatterdata such that each scatterplot shows the services correctly found at that cutoff
    #e.g. scatterplot 1 shows all services correctly found at cutoff 1, etc
    found_at_filter=("found_at",lambda actual,predictions,cutoff: actual==predictions[cutoff])
    #aggregate scatterdata such that each scatterplot shows the services found within the first cutoff positions of the predictions
    #e.g. scatterplot 5 shows all services correctly found at cutoff 1+2+3+4+5, etc
    found_within_filter=("found_within",lambda actual,predictions,cutoff: actual in predictions[0:cutoff+1])
    #aggregate scatterdata such that each scatterplot shows the services not yet found within the first cutoff positions of the predictions
    #e.g. scatterplot 5 shows all services not found at cutoff 1+2+3+4+5, etc
    not_found_within_filter=("not_found_within",lambda actual,predictions,cutoff: not actual in predictions[0:cutoff+1])

    (filter_name,filter_func)=not_found_within_filter
    max_cutoff=11

    cls = TemporalEvidencesClassifier(dataset.features,dataset.target_names)
    cls = cls.fit(dataset.data, dataset.target)
    results=cls.predict(dataset.data,include_conflict_theta=True)
    scatter_data=[]
    for cutoff in range(1,max_cutoff):
        scatter_data.append([(conflict,theta) for (actual,(predictions,conflict,theta)) in zip(dataset.target,results) 
                                              if cutoff-1<len(predictions) and filter_func(actual,predictions,cutoff-1)])
        
    plot.conflict_theta_scatter(scatter_data,base_filename="../plots/"+dataset.name+"/scatter/"+filter_name+"_")    
    #print json.dumps({"not_found_within_2":scatter_data[1],"not_found_within_4":scatter_data[3]})
    print json.dumps({"found_within_1":scatter_data[0]})

#dataset=load_mavlab()

dataset=load_kasteren("houseA")   
#write_evidences(dataset)
scatter_conflict_theta(dataset)
#histogram_compare_methods(dataset)
#confusion_matrix(dataset)
#timeline(dataset)
