import numpy
import pandas


def results_as_dataframe(true_targets, predicted_targets):
    """
    Converts the recommendation results into a pandas dataframe for easier evaluation.
    @param true_targets: A list of the actually performed user actions.
    @param predicted_targets: For each of the performed actions the list of calculated service recommendations.
    @return: A pandas dataframe that has as index the performed user actions (there is one row per action). The first
    column contains for each action the highest scoring recommendation, the second column contains the second best
    recommendation etc.
    """
    results = pandas.DataFrame(predicted_targets, index=pandas.Index(true_targets, name="Actual action"))
    results.columns = ["Rec. %d" % (r+1) for r in range(len(results.columns))]
    #results = results[results.columns[0:5]]
    return results


class AccuracyMetricsCalculator():
    #todo documentation of class

    def __init__(self, targets, recommendations):
        self.results = results_as_dataframe(targets, recommendations)

    def __unique_targets__(self): #todo
        occurring_actions = set(self.results.index.values)
        occurring_services = pandas.melt(self.results).dropna()["value"]
        occurring_services = set(occurring_services.unique())
        return sorted(occurring_actions | occurring_services)

    def true_positives(self, target):
        """
        Counts how often the given target was recommended correctly (true positives, TP).
        @param target: The name of the user action for which to count true positives.
        @return: A pandas dataset with column TP and several rows, first row lists #TP when one service was recommended,
        second row lists #TP when two services where recommended, etc.
        """
        #get all rows where the actual action corresponds to the given target
        r = self.results[self.results.index == target]
        #if recommendation matches the target, set column to "1" (true positive), else set to "0" (false negative)
        r = r.applymap(lambda col: 1 if col == target else 0)
        #count how many true positives there are in each column
        r = r.sum()
        #if have a true positive for n-th recommendation, then also have true positive for n+1, n+2 etc
        #-> calculate cumulative sum
        r = pandas.DataFrame(r.cumsum(axis=0), columns=["TP"]).applymap(float)
        return r

    def false_negatives(self, target):
        """
        Counts how often the given target was not recommended correctly (false negatives, FN).
        @param target: The name of the user action for which to count false negatives.
        @return: A pandas dataset with column FN and several rows, first row lists #FN when one service was recommended,
        second row lists #FN when two services where recommended, etc.
        """
        #the amount of false negatives corresponds to the difference between the total number of occurrences of the
        #target and the number of false positives
        total_occurrences = len(self.results[self.results.index == target])
        r = self.true_positives(target)
        r = r.apply(lambda tp: total_occurrences - tp)
        r.columns = ["FN"]
        return r

    def false_positives(self, target):
        """
        Counts how often the given target was recommended even though it didn't occur (false positives, FP).
        @param target: The name of the user action for which to count false positives.
        @return: A pandas dataset with column FP and several rows, first row lists #FP when one service was recommended,
        second row lists #FP when two services where recommended, etc.
        """
        #get all rows where the actual service does NOT correspond to the given target
        r = self.results[self.results.index != target]
        #if recommendation matches the target, set column to "1" (false positive), else set to "0" (true negative)
        r = r.applymap(lambda col: 1 if col == target else 0)
        #count how many false positives there are in each column
        r = r.sum()
        #if have a false positive for n-th recommendation, then also have false positive for n+1, n+2 etc
        #-> calculate cumulative sum
        r = pandas.DataFrame(r.cumsum(axis=0), columns=["FP"]).applymap(float)
        return r

    @staticmethod
    def precision(counts):
        """
        Calculate the precision as (true positives)/(true positives + false positives).
        @param counts: A dataframe that contains a column "TP" with true positives and "FP" with false positives.
        @return: A pandas dataframe with one column "Precision". The first row lists the achieved precision when only
        one recommendation is shown, the second row the precision when two recommendations are shown, etc.
        """
        p = counts["TP"]/(counts["TP"] + counts["FP"])
        p = pandas.DataFrame({"Precision": p}).fillna(0.0)
        return p

    @staticmethod
    def recall(counts):
        """
        Calculate the recall as (true positives)/(true positives + false negatives).
        @param counts: A dataframe that contains a column "TP" with true positives and "FN" with false negatives.
        @return: A pandas dataframe with one column "Recall". The first row lists the achieved recall when only
        one recommendation is shown, the second row the recall when two recommendations are shown, etc.
        """
        p = counts["TP"]/(counts["TP"] + counts["FN"])
        p = pandas.DataFrame({"Recall": p}).fillna(0.0)
        return p

    @staticmethod
    def f1(metrics):
        """
        Calculate the F1 as the harmonic mean of precision and recall.
        @param metrics: A dataframe with a column "Precision" and a column "Recall"
        @return:  A pandas dataframe with one column "F1". The first row lists the achieved F1 when only
        one recommendation is shown, the second row the F1 when two recommendations are shown, etc.
        """
        f = (2.0*metrics["Precision"]*metrics["Recall"]) / (metrics["Precision"]+metrics["Recall"])
        f = pandas.DataFrame({"F1": f}).fillna(0.0)
        return f

    def calculate_for_target(self, target):
        """
        Calculate precision, recall and F1 for one target (= one possible user action)
        @param target: Which user action to calculate the metrics for.
        @return: A pandas dataframe containing three columns "Precision", "Recall", "F1". The first row lists calculated
        metrics when only one recommendation is shown, the second row when two recommendations are shown etc,
        """
        #count how many true positives, false positives and false negatives occurred for this target
        counts = pandas.concat([self.true_positives(target),
                               self.false_negatives(target),
                               self.false_positives(target)],
                               axis=1)

        #use these counts to calculate the relevant metrics
        metrics = pandas.concat([self.precision(counts),
                                self.recall(counts)],
                                axis=1)
        metrics["F1"] = self.f1(metrics)["F1"]

        #add name of the target to the index, to prepare for merging the metrics for all targets
        metrics.index = pandas.MultiIndex.from_arrays([[target]*len(metrics),
                                                      [(i+1) for i in range(len(metrics.index))]],
                                                      names=["target", "cutoff"])

        return metrics

    def calculate(self):
        """
        Performs the actual calculation of the weighted average of precision, recall and F1 over all targets.
        @return: A pandas dataframe containing three columns "Precision", "Recall", "F1". The first row lists calculated
        metrics when only one recommendation is shown, the second row when two recommendations are shown etc
        """
        #make one big matrix with the metrics for all targets
        targets = self.__unique_targets__()
        metrics = pandas.concat([self.calculate_for_target(target) for target in targets])

        #count for each target how often the corresponding action actually occurred
        occurrences = pandas.TimeSeries(self.results.index.values).value_counts()
        occurrences = occurrences.reindex(targets).fillna(0).apply(float)

        #calculate the weighted average for each of the metrics
        #e.g.: weighted f1=((f1_target1*occurrences_target1)+...+f1_targetn*occurrences_targetn))/sum(occurrences)
        fix_index = lambda g: g.reset_index().drop("cutoff", axis=1).set_index("target").reindex(targets).fillna(0.0)  #todo
        weighted_average_for_column = lambda col: numpy.average(col.values, weights=occurrences.values)
        weighted_average = lambda group: fix_index(group).apply(weighted_average_for_column)
        metrics = metrics.groupby(level="cutoff").aggregate(weighted_average)

        return metrics


"""
def num_predictions(true_targets,predicted,max_cutoff):
    nums=[]
    for c in range(0,max_cutoff):
        n=[]
        for pred in predicted:
            n.append(min(c+1,len(pred)))
        nums.append(numpy.mean(numpy.array(n)))
    return nums

def confusion_matrix(true_targets,predicted_targets):
    labels=unique_labels(true_targets,predicted_targets,1)
    matrix = numpy.zeros(shape=(len(labels),len(labels)),dtype=numpy.int)
    for (actual,predictions) in zip(true_targets,predicted_targets):
        pred=predictions[0]
        matrix[labels[actual]][labels[pred]]+=1
    return (sorted(labels.keys()),matrix)
"""
 
                
               
