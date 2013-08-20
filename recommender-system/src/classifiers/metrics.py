from collections import defaultdict

import numpy

precision = lambda tp,fp: 0.0 if tp+fp==0.0 else float(tp)/float(tp+fp)
recall = lambda tp,fn: 0.0 if tp+fn==0.0 else float(tp)/float(tp+fn)
f1 = lambda p,r : 0.0 if p+r==0.0 else 2.0*p*r/(p+r)

def unique_labels(true_targets,predicted_targets,cutoff):
    labels=set(true_targets)
    for predicted in predicted_targets:
        labels |=set(predicted[0:cutoff])
    labels=sorted(list(labels))
    return dict((l,labels.index(l)) for l in labels)

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

 
                
               
