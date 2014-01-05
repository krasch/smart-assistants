import numpy
import pandas

precision = lambda tp,fp: 0.0 if tp+fp==0.0 else float(tp)/float(tp+fp)
recall = lambda tp,fn: 0.0 if tp+fn==0.0 else float(tp)/float(tp+fn)
f1 = lambda p,r : 0.0 if p+r==0.0 else 2.0*p*r/(p+r)

def unique_labels(true_targets,predicted_targets,cutoff):
    labels=set(true_targets)
    for predicted in predicted_targets:
        labels |=set(predicted[0:cutoff])
    labels=sorted(list(labels))
    return dict((l,labels.index(l)) for l in labels)



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


def true_positives(results, target):
    """
    Counts how often the given target was recommended correctly (true positives, TP) by a classifier.
    @param results: The results of running a classifier on the dataset, formatted as pandas dataset as described in
    `results_as_dataframe()`.
    @param target: The name of the user action for which to count true positives.
    @return: A pandas dataset with one column TP and several rows, first row lists #TP when one service was recommended,
    second row lists #TP when two services where recommended, etc.
    """
    #get all rows where the actual action corresponds to the given target
    r = results[results.index == target]
    #if recommendation matches the target, set column to "1" (true positive), else set to "0" (false negative)
    r = r.applymap(lambda col: 1 if col == target else 0)
    #count how many true positives there are in each column
    r = r.sum()
    #if have a true positive for n-th recommendation, then also have true positive for n+1, n+2 etc
    #-> calculate cumulative sum
    r = pandas.DataFrame(r.cumsum(axis=0), columns=["TP"]).applymap(float)
    return r


def false_negatives(results, target):
    """
    Counts how often the given target was not recommended correctly (false negatives, FN) by a classifier.
    @param results: The results of running a classifier on the dataset, formatted as pandas dataset as described in
    `results_as_dataframe()`.
    @param target: The name of the user action for which to count false negatives.
    @return: A pandas dataset with one column FN and several rows, first row lists #FN when one service was recommended,
    second row lists #FN when two services where recommended, etc.
    """
    #the amount of false negatives corresponds to the difference between the total number of occurrences of the
    #target and the number of false positives
    total_occurrences = len(results[results.index == target])
    r = true_positives(results, target)
    r = r.apply(lambda tp: total_occurrences - tp)
    r.columns = ["FN"]
    return r


def false_positives(results, target):
    """
    Counts how often the given target was recommended even though it dit not occur (false positives, FP) by a classifier.
    @param results: The results of running a classifier on the dataset, formatted as pandas dataset as described in
    `results_as_dataframe()`.
    @param target: The name of the user action for which to count false positives.
    @return: A pandas dataset with one column FP and several rows, first row lists #FP when one service was recommended,
    second row lists #FP when two services where recommended, etc.
    """
    #get all rows where the actual service does NOT correspond to the given target
    r = results[results.index != target]
    #if recommendation matches the target, set column to "1" (false positive), else set to "0" (true negative)
    r = r.applymap(lambda col: 1 if col == target else 0)
    #count how many false positives there are in each column
    r = r.sum()
    #if have a false positive for n-th recommendation, then also have false positive for n+1, n+2 etc
    #-> calculate cumulative sum
    r = pandas.DataFrame(r.cumsum(axis=0), columns=["FP"]).applymap(float)
    return r


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


def f1(metrics):
    """
    Calculate the f1 as the harmonic mean of precision and recall.
    @param metrics: A dataframe with a column "Precision" and a column "Recall"
    @return:  A pandas dataframe with one column "F1". The first row lists the achieved F1 when only
    one recommendation is shown, the second row the F1 when two recommendations are shown, etc.
    """
    f = (2.0*metrics["Precision"]*metrics["Recall"]) / (metrics["Precision"]+metrics["Recall"])
    f = pandas.DataFrame({"F1": f}).fillna(0.0)
    return f


def accuracy_metrics_for_target(results, target):
    """
    Calculate precision, recall and F1 for one target (= one actual user action)
    @param results: The
    @param target:
    @return:
    """

    counts = pandas.concat([true_positives(results, target),
                           false_negatives(results, target),
                           false_positives(results, target)],
                           axis=1)

    metrics = pandas.concat([precision(counts),
                            recall(counts)],
                            axis=1)
    metrics["F1"] = f1(metrics)["F1"]

    return metrics


def accuracy_metrics(true_targets, predicted_targets):
    r = results_as_dataframe(true_targets, predicted_targets)
    print r["Rec. 1"].unique()
    #print accuracy_metrics_for_target(r, "Hall-Bedroom_door=Closed")


def count_tpfpfn(true_targets,predicted_targets,cutoff):
    labels=unique_labels(true_targets,predicted_targets,cutoff)
    weights = [0]*len(labels)
    tp = [0]*len(labels)
    fp = [0]*len(labels)
    fn = [0]*len(labels)
    for i in range(len(true_targets)):
        true=true_targets[i]
        predicted=predicted_targets[i]
        weights[labels[true]]+=1
        if true in predicted[0:cutoff]:
           tp[labels[true]]+=1
        else:
           fn[labels[true]]+=1
        for pred in set(predicted[0:cutoff])-set([true]):
           fp[labels[pred]]+=1
    return (sorted(labels),tp,fp,fn,weights)

def multiple_predictions_scores_for_cutoff(true_targets,predicted_targets,cutoff):
    labels,tp,fp,fn,weights=count_tpfpfn(true_targets,predicted_targets,cutoff)
    precisions=numpy.array([precision(tp[i],fp[i]) for i in range(len(weights))])
    precision_average=numpy.average(precisions,weights=weights)
    recalls=numpy.array([recall(tp[i],fn[i]) for i in range(len(weights))])
    recall_average=numpy.average(recalls,weights=weights)
    f1s=numpy.array([f1(precisions[i],recalls[i]) for i in range(len(weights))])
    f1_average=numpy.average(f1s,weights=weights)
    return (precision_average,recall_average,f1_average)
    
 
def multiple_predictions_scores(true_targets,predicted,max_cutoff):
    precisions=[]
    recalls=[]
    f1s=[]
    for c in range(1,max_cutoff):
        (precision,recall,f1)=multiple_predictions_scores_for_cutoff(true_targets,predicted,c) 
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)    
    return (precisions,recalls,f1s)

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

 
                
               
