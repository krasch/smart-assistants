from datetime import datetime
from collections import defaultdict
from math import sqrt

from sklearn.cross_validation import KFold
import numpy
from scipy import stats

from data.kasteren import load as load_kasteren
from classifiers.randomc import RandomClassifier
from classifiers.bayes import NaiveBayesClassifier
from classifiers.temporal import TemporalEvidencesClassifier
from classifiers import metrics
import plot

def avg_std_CI(data):
 
    def CI(data):
        alpha=0.1
        t=stats.t.isf(alpha/2.0,len(data)-1)
        w=t*numpy.std(data,ddof=1)/sqrt(len(data))
        return w

    data=numpy.array(data)
    return (numpy.mean(data),numpy.std(data),CI(data))

class Results:

      def __init__(self):
          self.times={"train_time":[],"test_time":[],"test_time_ind":[]}
          self.metrics={"precision":defaultdict(list),"recall":defaultdict(list),"f1":defaultdict(list),"num_predictions":defaultdict(list)}
 

   
      def append_times(self,train_time,test_time,test_time_ind):
          self.times["train_time"].append(train_time)
          self.times["test_time"].append(test_time)
          self.times["test_time_ind"].append(test_time_ind)

      def append_metrics(self,(precision,recall,f1),num_predictions):
          for c in range(len(precision)):
              self.metrics["precision"][c].append(precision[c])
              self.metrics["recall"][c].append(recall[c])
              self.metrics["f1"][c].append(f1[c])
              self.metrics["num_predictions"][c].append(num_predictions[c])

      def __get_data(self,measure):
          if measure in self.times:
             return [self.times[measure]]
          elif measure in self.metrics:
             return [self.metrics[measure][c] for c in range(0,len(self.metrics[measure]))]
          raise RuntimeError("No measure %s defined"%measure)

      def summarize(self,measure):
          data=self.__get_data(measure)
          summarized=[avg_std_CI(d) for d in data]
          if measure in self.times:
             return summarized[0]
          return summarized

class Experiment:

      def __init__(self,dataset):
          self.dataset=dataset
          self.classifiers=[]
          self.results=dict()
  
      def add_classifier(self,cls,name=None):
          if name is None:
             name=cls.name
          self.classifiers.append((name,cls))

      def run_with_classifier(self,kf,cls):
          results=Results()
          for train,test in kf:
              #cv data
              data_train, data_test, target_train, target_test = self.dataset.data[train], self.dataset.data[test], self.dataset.target[train], self.dataset.target[test]
              #train
              train_time=datetime.now()
              cls = cls.fit(data_train, target_train)
              train_time=(datetime.now()-train_time).microseconds/1000.0
              #test
              test_time=datetime.now()
              predicted=cls.predict(data_test)
              test_time=(datetime.now()-test_time).microseconds/1000.0 
              #measures
              results.append_times(train_time,test_time,test_time/float(len(data_test)))
              results.append_metrics(metrics.multiple_predictions_scores(target_test,predicted,20),metrics.num_predictions(target_test,predicted,20))
          return results

      def run(self,folds):
          kf = KFold(len(self.dataset.data), n_folds=folds, indices=False)
          self.results=dict()
          for (name,cls) in self.classifiers:
             self.results[name]=self.run_with_classifier(kf,cls)

      def get_results(self,measures):
          results=dict()
          for m in measures:
              results[m]=dict((name,self.results[name].summarize(m)) for (name,cls) in self.classifiers)
          return results

      def plot_results(self,measures):
          for measure in measures:
              plot.plot_result(self.classifiers,self.results,measure,filename="../plots/"+self.dataset.name+"/"+measure+".pdf")

      def print_results(self,measures):
          for m in measures:
              print m
              summarized=dict((name,self.results[name].summarize(m)) for (name,cls) in self.classifiers)
              tabwidth=max([len(name) for name in summarized])+3
              print "   "+" ".join([name.rjust(tabwidth) for (name,cls) in self.classifiers])
              format_CI=lambda (avg,std,CI) : ("%.2f %.2f"%(avg,CI)).rjust(tabwidth) 
              max_cutoff=[len(s) for s in summarized.values()][0]
              for c in range(max_cutoff):
                  print str(c+1).ljust(3)+" ".join([format_CI(summarized[name][c]) for (name,cls) in self.classifiers])
              print  


      def print_results_latex(self,measures,cutoff=1):
          format_CI=lambda (avg,std,CI) : "$%.2f \pm %.2f$"%(avg,CI)
          for (name,cls) in self.classifiers:
              print name+" & "+" & ".join([ format_CI(self.results[name].summarize(m)[cutoff-1]) for m in measures])+"\\\\"
     
      def print_runtimes_latex(self,measures):  
          format_CI=lambda (avg,std,CI) : "$%.2f \pm %.2f$"%(avg,CI)
          for (name,cls) in self.classifiers:
              print name+" & "+" & ".join([ format_CI(self.results[name].summarize(m)) for m in measures])+"\\\\"        

