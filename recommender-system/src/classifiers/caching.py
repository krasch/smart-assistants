class CacheMiss(Exception):
    def __init__(self,set_features):
        self.set_features=set_features

    def __str__(self):
        return str(self.set_features)


class NoCache:

    def set_sources(self,sources,binning_method):
        pass

    def get_cached_evidences(self,set_features):
        raise CacheMiss(set_features)

    def update(self,set_features,masses,conflict,theta):
        pass


class PreviousItemCache:
    def __init__(self):
        self.cached_hash=None
        self.cached_results=None
        self.hits_counter=0
        self.misses_counter=0

    def __calculate_hash(self,set_features):
        hash_=[]
        for (attribute,value,timedelta) in sorted(set_features,key=lambda tup: tup[2]):
            hash_.append("%s=%s(%s)"%(attribute,value,str(self.key(timedelta))))
        return hash("".join(hash_))

    def set_sources(self,sources,binning_method):
        self.sources=sources
        self.key=binning_method.key

    def get_cached_evidences(self,set_features):
        hash_=self.__calculate_hash(set_features)
        if not self.cached_hash is None and self.cached_hash==hash_:
           self.hits_counter+=1
           return self.cached_results
        self.misses_counter+=1
        raise CacheMiss(set_features)
       

    def update(self,set_features,masses,conflict,theta):
        self.cached_hash=self.__calculate_hash(set_features)
        self.cached_results=(masses,conflict,theta)

    def get_statistics(self):
        return self.hits_counter,self.misses_counter

