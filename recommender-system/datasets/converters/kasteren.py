import argparse
from datetime import datetime,timedelta

from scipy.io import loadmat
from pandas import DataFrame

#map 0/1 to sensible sensor values
def get_sensor_value(sensor,setting):
    doors=["door","cupboard","freezer","fridge","dishwasher","washingmachine"]
    for d in doors:
        if d in sensor.lower():
           return ("Closed","Open")[setting]
    return ("Off","On")[setting]

def extract_from_matlab(matlab_file,house):

    convert_date=lambda d: datetime.fromordinal(int(d)) + timedelta(days=d%1) - timedelta(days = 366)
    convert_name=lambda n: n.replace(" ","_")

    #returns list of (time,sensor,sensor_value)
    def extract_sensor_data(data_array,sensors):
        sensor_data=[]
        for d in data_array:
            start,end,sensor_id,_=convert_date(d[0]),convert_date(d[1]),int(d[2]),int(d[3])
            if sensor_id in sensors:
               sensor=sensors[sensor_id]
               sensor_data.append((start,sensor,get_sensor_value(sensor,1)))
               sensor_data.append((end,sensor,get_sensor_value(sensor,0))) 
            else:
               print "Unknown sensor %d"%sensor_id
        return sensor_data

    #returns list of (start,end,activity_performed)
    def extract_activity_data(data_array,activities):
        activity_data=[] 
        for d in data_array:
            start,end,activity_id=convert_date(d[0]),convert_date(d[1]),int(d[2])
            if activity_id in activities:
                activity_data.append((start,end,activities[activity_id]))
            else:
                print "Unknown activity %d"%activity_id
        return activity_data 

    houseA=loadmat(matlab_file)[house][0][0]
    sensors={s[0][0][0]:convert_name(s[1][0]) for s in houseA[3]}
    activities={i+1:convert_name(a[0][0])  for (i,a) in enumerate(houseA[4]) if len(a[0])>0}
    sensor_data=extract_sensor_data(houseA[1][0][0][0],sensors)
    activity_data=extract_activity_data(houseA[2][0][0][0],activities)
    return sensors,sensor_data,activities,activity_data

def write_sensor_data_csv(house,sensor_data):
    data=DataFrame(sensor_data,columns=["timestamp","sensor","value"])
    data=data.set_index("timestamp")
    data=data.sort()
    data.to_csv("%s.csv"%house)

def convert_house(matlab_file,house):
    sensors,sensor_data,activities,activity_data=extract_from_matlab(args.datafile,house)
    write_sensor_data_csv(house,sensor_data) 
  
#location of the matlab file is given by command line argument
parser = argparse.ArgumentParser(description='Convert the Kasteren datasets into CSV files')
parser.add_argument("datafile", help="matlab file that contains the datasets")
args = parser.parse_args()

#convert houseA and houseB datasets to CSV files
convert_house(args.datafile,"houseA")
convert_house(args.datafile,"houseB")

