#TODO fix mavlab import

from datetime import datetime
from database import Database

def read_sensor_data(datafile):
    f = open(datafile)
    data=[]
    for line in f.readlines():
        if "AM" in line:
           time=line.split("AM")[0]+"AM"
           line=line.split("AM")[1].strip()
        elif "PM" in line:
           time=line.split("PM")[0]+"PM"
           line=line.split("PM")[1].strip()
        else:
           print "Bad line"
        time=datetime.strptime(time,"%m/%d/%Y %H:%M:%S %p")
        #time=datetime.strptime(time,"%b/%d/%Y %H:%M:%S")
        location=line.split(")")[0][1:]
        line=line.split(")")[1].strip()
        sensor=line.split(" ")[0]
        value=line.split(" ")[1]
        
        if len(data)>0:
           prev_time,prev_location,prev_sensor,prev_value=data[-1]
           if prev_time==time and prev_sensor==sensor and prev_value==value:
              continue
        data.append((time,location,sensor,value))
    print data
    return data

def write_to_database(data,database):
#2008-02-25 00:20:14.000000|Hall-Bedroom_door|Open
    database=Database(database)
    for (time,location,sensor,value) in data:
        database.add_event(time,sensor,value)
    database.commit()
    database.close()

data="april.in"
path_to_database="mavlab_april.db"

data=read_sensor_data(data)
write_to_database(data,path_to_database)
