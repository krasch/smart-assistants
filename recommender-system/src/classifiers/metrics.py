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

def accuracy_metrics(true_targets, predicted_targets, cutoff):

    def convert_to_dataframe():
        #index = pandas.MultiIndex.from_tuples(zip(range(len(true_targets)), true_targets),
        #                                      names=["ID", "Actual service"])
        results = pandas.DataFrame(predicted_targets, index = pandas.Index(true_targets, name="Actual action"))
        results.columns = ["Rec. %d" % (r+1) for r in range(len(results.columns))]
        results = results[results.columns[0:5]]
        return results

    results = convert_to_dataframe()


    def true_positives(target):
        #get all rows where the actual action corresponds to the given target
        r = results[results.index == target]
        #if recommendation matches the target, set column to "1" (true positive), else set to 0 (false negative)
        r = r.applymap(lambda col: 1 if col == target else 0)
        #count how many true positives there are in each column
        r = r.sum()
        #if have a true positive for n-th recommendation, then also have true positive for n+1, n+2 etc
        #-> calculate cumulative sum
        r = pandas.DataFrame(r.cumsum(axis=0), columns=["TP"])
        return r

    """ too complicated, to be removed
    def false_negatives(target):
        #get all rows where the actual service corresponds to the given target
        r = results.xs(target, level="Actual service")

        #given one row, there is at most one column X that matches the actual service (true positive). if more
        #recommendations where shown, they also include the correct service -> set X and all columns after X to "0"
        #if less recommendations where shown, the correct service was not included (false negative) -> set all
        #columns before X to "1".
        r = r.applymap(lambda col: 0 if col == target else numpy.nan) #find column X and set to "0", all others to "N/A"
        r = r.transpose().fillna(method="pad") #set all columns after X to "0"
        r = r.fillna(1).transpose() #set all columns before X to "1"

        #aggregate the counted false negatives and update labels
        r = pandas.DataFrame(r.sum(), columns=["FN"])
        print r
        return r
    """

    def false_negatives(target):
        #the amount of false negatives corresponds to the difference between the total number of occurrences of the
        #target and the number of false positives
        total_occurrences = len(results[results.index == target])
        r = true_positives(target)
        r = r.apply(lambda tp: total_occurrences - tp)
        r.columns = ["FN"]
        return r


    def false_positives(target):
        #get all rows where the actual service does NOT correspond to the given target
        r = results.xs(target, level="Actual service")

    counts = pandas.concat([true_positives("Hall-Bedroom_door=Closed"),
                           false_negatives("Hall-Bedroom_door=Closed")],
                           axis=1)
    print counts

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

 
                
               
