import numpy
import pandas

runtime_metrics = ["Training time", "Overall testing time", "Individual testing time"]
quality_metrics = ["Recall", "Precision", "F1", "# of recommendations"]


def results_as_dataframe(user_actions, recommendations):
    """
    Converts the recommendation results into a pandas dataframe for easier evaluation.
    @param user_actions: A list of the actually performed user actions.
    @param recommendations: For each of the performed actions the list of calculated service recommendations.
    @return: A pandas dataframe that has as index the performed user actions (there is one row per action). The first
    column contains for each action the highest scoring recommendation, the second column contains the second best
    recommendation etc.
    """
    results = pandas.DataFrame(recommendations, index=pandas.Index(user_actions, name="Actual action"))
    results.columns = [(r+1) for r in range(len(results.columns))]
    return results


class QualityMetricsCalculator():
    """
    This is a utility class that contains a number of methods for calculating overall quality metrics for the produced
    recommendations. In general these methods produce pandas dataframes with several rows, where each row corresponds
    to one "cutoff" point. For example, a cutoff "4" means that the system cuts the number of recommendations at four,
    i.e. the user is shown at most four recommendations. If some post-processing method was used (e.g. show fewer
    recommendations if the recommendation conflict is low), then it can happen that fewer than four recommendations
    are shown. For reference, the column "# of recommendations" lists the average of the number of recommendations that
    were actually shown to the user.
    """

    def __init__(self, actual_actions, recommendations):
        """
        Initialize the calculation of the quality metrics..
        @param actual_actions: A list of strings, each representing one actual user action.
        @param recommendations: A list of lists of strings with the same length as actual_actions. Each list of
        strings contains the calculated recommendations for the corresponding actual user action.
        @return:
        """
        self.results = results_as_dataframe(actual_actions, recommendations)

    def __unique_actions__(self):
        """
        It can happen that one potential user action never happened, but that the corresponding service was recommended.
        To be able to count these false positives, we must calculate the list of all potential actions.
        @return:
        """
        occurring_actions = set(self.results.index.values)
        occurring_services = pandas.melt(self.results).dropna()["value"]
        occurring_services = set(occurring_services.unique())
        return sorted(occurring_actions | occurring_services)

    def true_positives(self, action):

        """
        Counts how often the given action was recommended correctly (true positives, TP).
        @param action: The name of the user action for which to count true positives.
        @return: A pandas dataset with column TP and several rows, first row lists #TP at cutoff "1", the second row at
        cutoff "2", etc.
        """
        #get all rows where the actual action corresponds to the given action
        r = self.results[self.results.index == action]

        if len(r) == 0:
            #if there are no such rows, then we have zero true positives, fill result dataframe with zeroes
           true_positives = pandas.Series(0.0, index=self.results.columns)
        else:
            #if recommendation matches the action, set column to "1" (true positive), else set to "0" (false negative)
            r = r.applymap(lambda col: 1 if col == action else 0).fillna(0)
            #count how many true positives there are in each column
            r = r.sum()
            #if have a true positive for n-th recommendation, then also have true positive for n+1, n+2 etc
            #-> calculate cumulative sum
            true_positives = r.cumsum(axis=0).apply(float)

        true_positives = pandas.DataFrame(true_positives, columns=["TP"])
        true_positives.index.name = "cutoff"
        return true_positives

    def true_positives_for_all(self):
        """
        Create a matrix that contains information about true positives for all possible actions.
        @return: A pandas with one column for each action, first row lists #TP at cutoff "1", the second row at
        cutoff "2", etc.
        """
        tp = [self.true_positives(action) for action in self.__unique_actions__()]
        tp = pandas.concat(tp, axis=1)
        tp.columns = self.__unique_actions__()
        return tp

    def false_negatives(self, action):
        """
        Counts how often the given action was not recommended correctly (false negatives, FN).
        @param action: The name of the user action for which to count false negatives.
        @return: A pandas dataset with column FN and several rows, first row lists #FN cutoff "1", the second row at
        cutoff "2", etc.
        """
        #the amount of false negatives corresponds to the difference between the total number of occurrences of the
        #action and the number of false positives
        true_positives = self.true_positives(action)
        total_occurrences = len(self.results[self.results.index == action])
        total_occurrences = pandas.Series(total_occurrences, index = true_positives.index)
        false_negatives = total_occurrences - true_positives["TP"]

        false_negatives = pandas.DataFrame(false_negatives, columns=["FN"])
        false_negatives.index.name = "cutoff"
        return false_negatives

    def false_positives(self, action):
        """
        Counts how often the given action was recommended even though it didn't occur (false positives, FP).
        @param action: The name of the user action for which to count false positives.
        @return: A pandas dataset with column FP and several rows, first row lists #FP at cutoff "1", the second row at
        cutoff "2", etc.
        """
        #get all rows where the actual service does NOT correspond to the given action
        r = self.results[self.results.index != action]
        if len(r) == 0:
            #if there are no such rows, then we have zero false positives, fill result dataframe with zeroes
            false_positives = pandas.Series(0.0, index=self.results.columns)
        else:
            #if recommendation matches the action, set column to "1" (false positive), else set to "0" (true negative)
            r = r.applymap(lambda col: 1 if col == action else 0)
            #count how many false positives there are in each column
            r = r.sum()
            #if have a false positive for n-th recommendation, then also have false positive for n+1, n+2 etc
            #-> calculate cumulative sum
            false_positives = r.cumsum(axis=0).apply(float)

        false_positives = pandas.DataFrame(false_positives, columns=["FP"])
        false_positives.index.name = "cutoff"
        return false_positives

    @staticmethod
    def precision(counts):
        """
        Calculate the precision as (true positives)/(true positives + false positives).
        @param counts: A dataframe that contains a column "TP" with true positives and "FP" with false positives.
        @return: A pandas dataframe with one column "Precision". The first row lists the achieved precision at cutoff
        "1", the second row at cutoff "2", etc.
        """
        p = counts["TP"]/(counts["TP"] + counts["FP"])
        p = pandas.DataFrame({"Precision": p}).fillna(0.0)
        return p

    @staticmethod
    def recall(counts):
        """
        Calculate the recall as (true positives)/(true positives + false negatives).
        @param counts: A dataframe that contains a column "TP" with true positives and "FN" with false negatives.
        @return: A pandas dataframe with one column "Recall". The first row lists the achieved recall at cutoff "1",
        the second row at cutoff "2", etc.
        """
        p = counts["TP"]/(counts["TP"] + counts["FN"])
        p = pandas.DataFrame({"Recall": p}).fillna(0.0)
        return p

    @staticmethod
    def f1(metrics):
        """
        Calculate the F1 as the harmonic mean of precision and recall.
        @param metrics: A dataframe with a column "Precision" and a column "Recall"
        @return:  A pandas dataframe with one column "F1". The first row lists the achieved F1 at cutoff "1", the second
        row at cutoff "2", etc.
        """
        f = (2.0*metrics["Precision"]*metrics["Recall"]) / (metrics["Precision"]+metrics["Recall"])
        f = pandas.DataFrame({"F1": f}).fillna(0.0)
        return f

    def number_of_recommendations(self):
        """
        Count how many recommendations the user was actually shown (e.g. when using a dynamic cutoff such as "show
        less recommendations when recommendation conflict is low").Number of recommendation is not an quality metric
        but fits here conceptually.
        @return: A pandas dataframe with one column "# of recommendations". The first row lists the # at cutoff "1", the
        second row at cutoff "2", etc.
        """
        n = (self.results.count(axis=0)/float(len(self.results))).cumsum()
        n = pandas.DataFrame({"# of recommendations": n})
        n.index.name = "cutoff"
        return n

    def calculate_for_action(self, action):
        """
        Calculate precision, recall and F1 for one action (= one possible user action)
        @param action: Which user action to calculate the metrics for.
        @return: A pandas dataframe containing columns for "Precision", "Recall", "F1". The first row lists
        calculated metrics at cutoff "1", the second row at cutoff "2", etc. A fourth column "action" simply lists the
        action name in all rows, this column is necessary for later merging the metrics of all actions.
        """
        #count how many true positives, false positives and false negatives occurred for this action
        counts = pandas.concat([self.true_positives(action),
                               self.false_negatives(action),
                               self.false_positives(action)],
                               axis=1)

        #use these counts to calculate the relevant metrics
        metrics = pandas.concat([self.precision(counts),
                                self.recall(counts)],
                                axis=1)
        metrics["F1"] = self.f1(metrics)["F1"]

        #add column that contains name of the action in all rows, to prepare for merging the metrics for all actions
        metrics["action"] = pandas.Series(action, index=metrics.index)

        return metrics

    def calculate(self):
        """
        Performs the actual calculation of the weighted average of precision, recall and F1 over all actions and counts
        the number of recommendations that where actually shown to the user.
        @return: A pandas dataframe containing one column for each of the four quality metrics. The first row lists
        calculated metrics at cutoff "1", the second row at cutoff "2"
        """
        #make one big matrix with the metrics for all actions
        actions = self.__unique_actions__()
        metrics = pandas.concat([self.calculate_for_action(action) for action in actions])

        #count for each action how often the corresponding action actually occurred
        occurrences = pandas.TimeSeries(self.results.index.values).value_counts()
        occurrences = occurrences.reindex(actions).fillna(0)

        #calculate the weighted average for each of the metrics (i.e. actions that occur more often have a higher
        #influence on the overall results for "Precision", "Recall and "F1")
        actions_as_index = lambda group: group.set_index("action").reindex(actions).fillna(0.0)
        weighted_average_for_column = lambda col: numpy.average(col.values, weights=occurrences.values)
        weighted_average = lambda group: actions_as_index(group).apply(weighted_average_for_column)
        metrics = metrics.groupby(level="cutoff").aggregate(weighted_average)
        del(metrics["action"])  #get rid of now unnecessary "action" column

        #do not need weighted average for # of recommendations, simply add counts as fourth column
        metrics["# of recommendations"] = self.number_of_recommendations()
        return metrics

    def confusion_matrix(self):
        """
        Calculate a confusion matrix: for each action count how often each service was recommended
        @return: A pandas dataframe, with one row for each possible action and one row for each possible
        service recommendation. Each matrix item counts how often the service was recommended when the action happened.
        """
        cutoff = 1   #only makes sense for cutoff=1

        def confusions_for_action(action):
            r = self.results[self.results.index == action][cutoff]
            return r.groupby(r).count()

        actions = self.__unique_actions__()
        matrix = [confusions_for_action(action) for action in actions]
        matrix = pandas.concat(matrix, axis=1).reindex(actions).transpose()
        matrix.index = actions
        matrix = matrix.reindex(actions).fillna(0)
        matrix.index.name = "Actual action"

        return matrix




               
