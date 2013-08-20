from dataset import Dataset,dataset_to_scikit,dataset_to_arff,UnknownDataset

base_path="../datasets/"


def db_path(base_path,house):
    return base_path+house+".csv"    

def load(house,base_path=base_path):
    if house=="houseA":
       exclude_sensors=["activity"]
       exclude_services=["ToiletFlush=Off"]
    elif house=="houseB":
       exclude_sensors=["activity"]
       exclude_services=["ToiletFlush=Off","toilet_flush=Off","pressure_mat","PIR","mercury_switch_dresser_door=Closed",
                         "mercury_switch_cutlary_drawer=Off","mercurary_switch_stove_lid=Off",]
                        # "mercury_switch_dresser_door=Open"]
    else:
        raise UnknownDataset(house)
    dataset=Dataset(house)
    dataset.read(db_path(base_path,house),exclude_sensors=exclude_sensors,exclude_actions=exclude_services)
    return dataset

def load_scikit(house):
    
    dataset=load(house)
    scikit=dataset_to_scikit(dataset)
    return scikit

load("houseA")
