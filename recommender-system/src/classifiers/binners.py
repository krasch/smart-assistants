import numpy

#copied from numpy cookbook
#http://www.scipy.org/Cookbook/SignalSmooth

def smooth(x,window_len=10,window='hanning'):
    window_len=min(window_len,len(x)-1)
    if x.ndim != 1:
       raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
       raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
       return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
       raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
       w=numpy.ones(window_len,'d')
    else:  
       w=eval('numpy.'+window+'(window_len)')
    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]


"""
#bins the timedeltas according to a fixed list of bins
standard_bins=list(range(0,60,10))+list(range(60,300,30))
def fixed_bins_binning(bins=standard_bins):

    #returns a function that can look up the correct key for this specific binning
    def lookup_function(bins):
        def lookup(timedelta):
            if timedelta is None:
               raise LookupFailed(timedelta)
            for b in range(len(bins)):
               if timedelta<bins[b]:  
                   return b-1
            raise LookupFailed(timedelta)     
        return lookup        

    #perform the binning
    def perform(timedeltas,smoothing):
        lookup=lookup_function(bins)
        binned=[0]*len(bins)
        for t in timedeltas:
           try:
              key=lookup(t) 
              binned[key]+=1
           except LookupFailed:
              pass
        binned=list(smooth(numpy.array(binned[target])))
        return binned,lookup

    return perform
"""


class StaticBinning:

   standard_bins=list(range(0,60,10))+list(range(60,300,30))
    
   def __init__(self,bins=standard_bins):
       self.bins=bins

   def perform_binning(self,timedeltas):
       binned=[0]*len(self.bins)
       for t in timedeltas:
           key=self.key(t)
           if not key is None:
              binned[key]+=1
       return binned

   def key(self,timedelta):
       if timedelta is None:
          return None
       for b in range(len(self.bins)):
         if timedelta<self.bins[b]:  
            return b-1

def bin_timedeltas(counts,binning_method,smoothing=True):
    binned=dict()
    for (target,timedeltas) in counts.items():
        binned[target]=binning_method.perform_binning(timedeltas)
        if smoothing:
           binned[target]=list(smooth(numpy.array(binned[target])).clip(0.0000001))
    return binned  


