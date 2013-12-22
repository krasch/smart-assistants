import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def plot_evidences(classifier):
    bins=classifier.binning_method.bins
    for source in classifier.sources.values():
        observations_by_target=dict()
        for t,target in enumerate(classifier.target_names):
            observations_by_target[target]=[source.temporal_counts[b][t] for b in range(len(bins))]
        observed_sorted=sorted(observations_by_target,key=lambda k:sum(observations_by_target[k]),reverse=True)[0:5]  #sort by sum of counts
        for observed in observed_sorted:
            plt.plot(bins,observations_by_target[observed])
        plt.xlabel("Time since setting changed [seconds]")
        plt.ylabel("Observed actions (smoothed)")
        plt.legend(observed_sorted) 
        plt.savefig("../plots/"+source.source_name()+".pdf")
        plt.close()   

def plot_result(classifiers,results,measure,filename):
    plt.figure(figsize=(6,6),dpi=300)
    plt.ylabel(measure,fontsize=18)
    plt.xlabel('Number of recommendations',fontsize=18)    
    legend=[]
    colors=["#6dad22","#4e88d8","#db6d49","b","g","r","c","m","y"]
    markers=["o","d","*","^","o","o"]
    for (i,(cls_name,cls)) in enumerate(classifiers):
        y=[avg for (avg,std,CI) in results[cls_name].summarize(measure)]
        x=range(1,len(y)+1)
        plt.plot(x,y,color=colors[i],marker=markers[i],markersize=10)
        legend.append(cls_name)
    fontP = FontProperties()
    fontP.set_size(18)
    plt.legend(tuple(legend),loc=4,prop=fontP) 
    plt.xlim(1,14) 
    if not measure=="num_predictions":
       plt.ylim(0,1.0)
    plt.xticks(range(2,16,2),range(2,16,2), fontsize=18) 
    plt.savefig(filename)
    plt.close() 


def plot_accuracy_comparison(results, metric, filename):
    plt.figure(figsize=(6, 6), dpi=300)
    plt.ylabel(metric, fontsize=18)
    plt.xlabel('# of recommendations shown', fontsize=18)
    plt.plot(results)
    plt.legend(legend=results.columns.values, loc="best")
    if not metric == "# of recommendations":
        plt.ylim(0.0, 1.0)
    plt.savefig(filename)
    plt.close()

def plot_train_size(classifiers,sizes,results,xticks,measure,filename):
    plt.figure(figsize=(6,6),dpi=300)
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel(measure,fontsize=18)
    plt.xlabel('Size of training data',fontsize=18)   
    legend=[]
    colors=["#6dad22","#4e88d8","#db6d49","b","g","r","c","m","y"]
    markers=["o","d","*","^","o","o"]
    for (i,cls_name) in enumerate(classifiers):
        plt.plot(sizes,results[cls_name],color=colors[i],marker=markers[i],markersize=10)
        legend.append(cls_name)
    fontP = FontProperties()
    fontP.set_size(18)
    plt.legend(tuple(legend),loc=4,prop=fontP) 
    if not measure=="num_predictions":
       plt.ylim(0,1.0)
    plt.xlim(sizes[0],sizes[-1])
    if not xticks is None:
       plt.xticks(xticks[0],xticks[1]) 
    plt.savefig(filename)
    plt.close()  


def conflict_theta_scatter(data,base_filename):
    for (cutoff,d) in enumerate(data):
        plt.figure(figsize=(6,6),dpi=300)
        conflict_data=[conflict for (conflict,theta) in d]
        theta_data=[theta for (conflict,theta) in d]
        plt.plot(conflict_data,theta_data,'x',color="#6dad22")
        plt.xticks([0.0,0.5,1.0],[0.0,0.5,1.0], fontsize=18)
        plt.yticks([0.5,1.0],[0.5,1.0], fontsize=18)
        plt.xlabel("conflict",fontsize=18)
        plt.ylabel("uncertainty",fontsize=18)
        plt.xlim(0,1.0) 
        plt.ylim(0,1.0) 
        plt.savefig("%s%d.eps"%(base_filename,cutoff+1))
        plt.close()

def histogram(to_compare,data,filename):
    bar_width = 0.75       
    colors=["lightblue","lightgreen","lightcoral","orange","pink"]
    services=[service for (service,executed,correct) in data[to_compare[0]]]

    for (c_index,c) in enumerate(to_compare):
        x=range(c_index,len(services)*(len(to_compare)+1),len(to_compare)+1)  
        y=[executed for (service,executed,correct) in data[c]]       
        plt.bar(x,y,bar_width,color='grey',alpha=0.3)
        y=[correct for (service,executed,correct) in data[c]]       
        plt.bar(x,y,bar_width,color=colors[c_index])
    
    x=range(int(len(to_compare)/2),len(services)*(len(to_compare)+1),len(to_compare)+1)
    plt.xticks(x,services,rotation=90,size=9)
    plt.xlim(0,x[-1]+1) 
    plt.subplots_adjust(bottom=0.4)
    #plt.legend(tuple(to_compare),loc=0) 
    plt.savefig(filename)
    plt.close()   
