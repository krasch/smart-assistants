from dataset import Dataset,dataset_to_scikit,write_dataset_as_arff,UnknownDataset

database="../datasets/mavlab_april.db"

def load():
    exclude_sensors=[]
    exclude_services=[]
    dataset=Dataset("mavlab")
    dataset.read(database,exclude_sensors=exclude_sensors,exclude_actions=exclude_services)
    return dataset

def load_scikit():
    
    dataset=load()
    scikit=dataset_to_scikit(dataset)
    return scikit

