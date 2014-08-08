"""
Compare prediction quality and runtimes of several classifiers for one dataset. Details
for this experiment can be found in the paper in Section 6.4 and in the dissertation in Section 5.5.4,
"""
        
import sys
sys.path.append("..") 

from evaluation.experiment import Experiment
from evaluation import plot
from recsys.classifiers.randomc import RandomClassifier
from recsys.classifiers.bayes import NaiveBayesClassifier
from recsys.classifiers.temporal import TemporalEvidencesClassifier
from recsys.dataset import load_dataset
import config

def houseA():
    """ 
    This dataset has 14 binary sensors, i.e. at most 14 services are typically available concurrently. The only anomaly
    is right at the beginning of the dataset, where the current status of the sensors are not known. In this case more
    than 14 services can be recommended. However, there will be few instances where this is the case and
    recommendation results will be be statistically insignificant for these values. For this reason, when printing or
    plotting results, cut the recommendation results at 14 services.
    """
    data = load_dataset("../datasets/houseA.csv", "../datasets/houseA.config")
    cutoff_results_at = 14
    return data, cutoff_results_at
    
def houseB():   
    """
    This dataset is partially dominated by one of the sensors, which makes the evaluation results less statistically
    sound, e.g. it leads to large confidence intervals when running 10-fold cross-validation.  
    """
    data = load_dataset("../datasets/houseB.csv", "../datasets/houseB.config")
    cutoff_results_at = 15    
    return data, cutoff_results_at

#configuration
data, cutoff_results_at = houseA()

#run several classifiers on the same dataset, use 10-fold cross-validation
experiment = Experiment(data)
experiment.add_classifier(TemporalEvidencesClassifier(data.features, data.target_names), name="Our method")
experiment.add_classifier(NaiveBayesClassifier(data.features, data.target_names), name="Naive Bayes")
experiment.add_classifier(RandomClassifier(data.features, data.target_names), name="Random")
results = experiment.run(folds=10)

#print and plot results
results.print_quality_comparison_at_cutoff(cutoff=1, metrics=["Recall", "Precision", "F1"])
results.print_runtime_comparison()
plot_conf = plot.plot_config(config.plot_directory, sub_dirs=[data.name], img_type=config.img_type)
results.plot_quality_comparison(metrics=["Recall", "Precision", "F1"], plot_config=plot_conf,
                                cutoff_results_at=cutoff_results_at)
                                

