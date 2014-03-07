"""
Use this module to run the experiments described in:
- the accompanying paper "An unsupervised recommender system for smart homes" (add link when it article is online)
- my dissertation (http://www.diva-portal.org/smash/record.jsf?pid=diva2:650328)
Short descriptions of the experiments can also be found in `evaluation/paper_experiments.py`
Additional methods for examining a dataset and evaluating recommendation results can also be found in `examine.py`.
"""

import os

from evaluation.paper_experiments import *
from dataset import load_dataset_as_sklearn

#per default store plots in ../plots/
plot_directory = os.path.join(os.pardir, "plots")
img_type = "pdf"

"""
Select the dataset you want to use  by commenting/uncommenting.
"""
def initialize_houseA():
    dataset = load_dataset_as_sklearn("../datasets/houseA.csv", "../datasets/houseA.config")
    #This dataset has 14 binary sensors, i.e. at most 14 services are typically available concurrently. The only anomaly
    #is right at the beginning of the dataset, where the current status of the sensors are not known. In this case more
    #than 14 services can be recommended. However, there will be few instances where this is the case and
    #recommendation results will be be statistically insignificant for these values. For this reason, when printing or
    #plotting results, cut the recommendation results at 14 services.
    return PaperExperiments(dataset, cutoff_results_at=14,
                            plot_directory=plot_directory, img_type=img_type)

def initialize_houseB():
    #This dataset is partially dominated by one of the sensors, which makes the evaluation results less statistically
    #sound, e.g. it leads to large confidence intervals when running 10-fold cross-validation.
    dataset = load_dataset_as_sklearn("../datasets/houseB.csv", "../datasets/houseB.config")
    return PaperExperiments(dataset, cutoff_results_at=15,
                            plot_directory=plot_directory, img_type=img_type)

#evaluate using the Kasteren houseA dataset
experiment = initialize_houseA()
#evaluate using the Kasteren houseB dataset
#experiment = initialize_houseB()

"""
Select the experiments you want to run by commenting/uncommenting.
"""

#compare quality and runtimes of several classifiers for one dataset
experiment.compare_classifiers()

#evaluate how the proposed recommender system copes with varying interval widths and Delta t_max settings.
#experiment.evaluate_interval_settings()

#scatter conflict versus uncertainty to find regions of uncertainty/conflict where the algorithm is more/less successful
#the output can be found in the "../plots/<dataset_name>/scatter" directory
#experiment.scatter_conflict_uncertainty()

#evaluate the benefit of dynamic cutoff methods, i.e. show less recommendations if uncertainty and conflict are low
#experiment.evaluate_dynamic_cutoff()

#evaluate how the classifiers behave for different sizes of the training dataset
#experiment.evaluate_training_size()

#evaluate how the proposed recommendation algorithm scales for larger datasets using synthetic data
#scalability_experiment()