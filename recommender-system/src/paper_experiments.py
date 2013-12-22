from collections import defaultdict
import json

from sklearn.datasets.base import Bunch
import numpy

from data.kasteren import load_scikit as load_kasteren
from data.mavlab import load_scikit as load_mavlab
from experiment import Experiment
from classifiers.randomc import RandomClassifier
from classifiers.bayes import NaiveBayesClassifier
from classifiers.temporal import TemporalEvidencesClassifier
from classifiers.binners import StaticBinning
from classifiers.postprocess import dynamic_cutoff
from classifiers.caching import PreviousItemCache
import plot

def compare_classifiers(dataset):
    print "Compare classifiers for dataset "+dataset.name
    experiment=Experiment(dataset)
    #experiment.add_classifier(TemporalEvidencesClassifier(dataset.features,dataset.target_names),name="Our method")
    #experiment.add_classifier(TemporalEvidencesClassifier(dataset.features,dataset.target_names,cache=PreviousItemCache()),name="Cached")
    experiment.add_classifier(NaiveBayesClassifier(dataset.features, dataset.target_names), name="Naive Bayes")
    experiment.add_classifier(RandomClassifier(dataset.features, dataset.target_names), name="Random")
    experiment.run(folds=10)
    #experiment.print_runtime_comparison()
    #experiment.print_accuracy_comparison()
    experiment.plot_accuracy_comparison()

def compare_intervals(dataset):
    print "Comparing different intervals"
    experiment=Experiment(dataset)
    intervals_to_test=[#("bis 60 10, bis 300 30",list(range(0,60,10))+list(range(60,300,30))), 
                       #("bis 60 10, bis 600 30",list(range(0,60,10))+list(range(60,600,30))), 
                       ("$\Delta t_{\mathit{max}}=10$",list(range(0,10,10))),                             #shortshort max_t
                       ("$\Delta t_{\mathit{max}}=15$",list(range(0,15,10))), 
                       ("$\Delta t_{\mathit{max}}=30$",list(range(0,30,10))),                             #shortshort max_t
                       ("$\Delta t_{\mathit{max}}=60$",list(range(0,60,10))),                             #short max_t
                       ("$\Delta t_{\mathit{max}}=120$",list(range(0,60,10))+list(range(60,120,30))),   
                         #short max_t
                       #("$\Delta t_{\mathit{max}}=1200$",list(range(0,60,10))+list(range(60,1200,30))),   #long max_t
                       #("all intervals 2s wide",list(range(0,300,2))),
                       #("all intervals 4s wide",list(range(0,300,4))),
                       #("all intervals 6s wide",list(range(0,300,6))), 
                       #("all intervals 8s wide",list(range(0,300,6))),
                       #("all intervals 30s wide",list(range(0,300,30))),
                       #("all intervals 50s wide",list(range(0,300,50))),
                       #("all intervals 100s wide",list(range(0,300,100))),
                      ]
    for (name,bins) in intervals_to_test:
        experiment.add_classifier(TemporalEvidencesClassifier(dataset.features,dataset.target_names,
                                  binning_method=StaticBinning(bins=bins)),name=name)
    experiment.run(folds=10)
    print "recall,precision,f1"
    experiment.print_results_latex(["recall","precision","f1"])
    print "runtimes"
    experiment.print_runtimes_latex(["train_time","test_time","test_time_ind"])
    experiment.plot_results(["precision","recall"])

def compare_postprocess(dataset):
    print "Comparing postprocessing methods"
    experiment=Experiment(dataset)
    methods_to_test=[("Fixed cutoff",None),
                     ("dynamic cutoff=4",dynamic_cutoff(1.0,0.4,4)),
                     ("dynamic cutoff=2",dynamic_cutoff(1.0,0.4,2))]
    for (name,method) in methods_to_test:
        experiment.add_classifier(TemporalEvidencesClassifier(dataset.features,dataset.target_names,
                                  postprocess=method),name=name) 
    experiment.run(folds=10)
    experiment.plot_results(["precision","recall","num_predictions","f1"])
    experiment.print_results_latex(["num_predictions","recall","precision"],cutoff=5)
    experiment.print_results_latex(["num_predictions","recall","precision"],cutoff=10)

"""
#test size=training size
def training_size_experiment(dataset):
    def smaller_dataset(ratio_of_original):
        new_size=int(len(dataset.data)*ratio_of_original*2.0)
        return Bunch(name=dataset.name,
                 data=dataset.data[0:new_size],
                 target=dataset.target[0:new_size],
                 features=dataset.features,
                 times=dataset.times[0:new_size],
                 target_names=dataset.target_names)

    print "Evaluating training size"
    ratios=list(numpy.arange(0.02,0.5,0.05))
    results=[]
    for r in ratios:
        partial_data=smaller_dataset(r)
        experiment=Experiment(partial_data)
        experiment.add_classifier(TemporalEvidencesClassifier(dataset.features,dataset.target_names),name="Our method")
        experiment.add_classifier(NaiveBayesClassifier(dataset.features,dataset.target_names),name="Naive Bayes")     
        experiment.run(folds=2)
        #experiment.print_results_latex(["recall","precision","f1"])
        local_results=experiment.get_results(["f1"])["f1"]
        results.append((int(len(partial_data.data)/2.0),local_results["Our method"][0][0],local_results["Naive Bayes"][0][0]))
    print results
"""

#test size=total size-training size
def training_size_experiment(dataset):
    def calculate_training_time(last_train):
        delta=dataset.times[last_train]-dataset.times[0]
        time_seconds=delta.days*24*60*60+delta.seconds
        return time_seconds/float(24*60*60)

    experiment=Experiment(dataset)
    experiment.add_classifier(TemporalEvidencesClassifier(dataset.features,dataset.target_names),name="Our method")
    experiment.add_classifier(NaiveBayesClassifier(dataset.features,dataset.target_names),name="Naive Bayes")
    experiment.add_classifier(RandomClassifier(dataset.features,dataset.target_names),name="Random")
    
    train_sizes=[10,25,50,75]+[int(r*len(dataset.data)) for r in list(numpy.arange(0.05,1.00,0.05))]
    train_times=[calculate_training_time(train_size) for train_size in train_sizes]
    #print train_times
    #length_times=[dataset.data[100].
    results=defaultdict(list)
    for train_size,train_time in zip(train_sizes,train_times):
        test_size=len(dataset.data)-train_size
        train_data=numpy.array([True]*train_size+[False]*test_size)
        #test_data=numpy.array([False]*train_size+[True]*test_size)
        test_data=numpy.array([True]*len(dataset.data))
        print train_size,train_time
        for cls_name,cls in experiment.classifiers:
            r=experiment.__run_with_classifier__([(train_data,test_data)],cls)
            results[cls_name].append(r.summarize("f1")[0][0])


    print results
    xticks=None
    if dataset.name=="houseA":
       counts=[500,1000,1500,2000]
       labels=["%i\n(%i days)"%(count,round(calculate_training_time(count))) for count in counts]
       xticks=counts,labels
    elif dataset.name=="houseB":
       counts=[500,1000,1500,2000,2500] 
       labels=["%i\n(%i days)"%(count,round(calculate_training_time(count))) for count in counts]
       xticks=counts,labels
    plot.plot_train_size([cls_name for cls_name,cls in experiment.classifiers],train_sizes,results,xticks,"f1","../plots/%s/train_size.pdf"%dataset.name)
    plot.plot_train_size([cls_name for cls_name,cls in experiment.classifiers],train_times,results,None,"f1","../plots/%s/train_time.pdf"%dataset.name)


    #print json.dumps({"train_size":train_sizes,"F1":results,"xticks":xticks},sort_keys=True,indent=4, separators=(',', ': '))
    
 
    

#mavlab=load_mavlab()
#print len(mavlab.features)
#print len(mavlab.data)
house=load_kasteren("houseA")
compare_classifiers(house)
#compare_intervals(house)
#compare_postprocess(houseA)
#training_size_experiment(house)
