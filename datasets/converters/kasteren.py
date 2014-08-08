import argparse
from datetime import datetime,timedelta

from scipy.io import loadmat
from pandas import DataFrame


#convert matlab time to python datetime, matlab time is off 366 days
to_datetime = lambda d: datetime.fromordinal(int(d)) + timedelta(days=d % 1) - timedelta(days= 366)


#obtain data for houseA or houseB from matlab data,
#lots of obscure references to array position where data resides in matlab file
def extract_raw_data(matlab_data):
    extract_sensors = lambda data: {s[0][0][0]: s[1][0]for s in data[3]}
    extract_activities = lambda data: {i+1: a[0][0] for (i, a) in enumerate(data[4]) if len(a[0]) > 0}
    extract_sensor_data = lambda data: [(s[0], s[1], s[2], s[3]) for s in data[1][0][0][0]]
    extract_activity_data = lambda data : [(a[0], a[1], a[2]) for a in data[2][0][0][0]]

    matlab_data = matlab_data[0][0]
    return {"sensors": extract_sensors(matlab_data),
            "activities": extract_activities(matlab_data),
            "sensor_data": extract_sensor_data(matlab_data),
            "activity_data": extract_activity_data(matlab_data)
    }


def format_sensor_data(raw_sensor_data, sensors):

    #convert matlab timestamp to timedelta, convert sensor ids to integer
    initial_formatting = lambda (start, end, sid, status): (to_datetime(start), to_datetime(end), int(sid))

    #identify data from unknown sensors and remove from data
    def remove_unknown_sensors(data):
        sensor_exists = lambda (start, end, sid) : sid in sensors
        sensor_id_to_name = lambda (start, end, sid) : (start, end, sensors[sid])

        unknown_sensors = [d for d in data if not sensor_exists(d)]
        unknown_sensors = [sid for start, end, sid in unknown_sensors]
        unknown_sensors = ["Unknown sensor %d occurred %d times" % (sid, unknown_sensors.count(sid))
                           for sid in set(unknown_sensors)]
        print "\n".join(unknown_sensors)

        data = [sensor_id_to_name(d) for d in data if sensor_exists(d)]
        return data

    #map 0/1 to sensible status names
    def human_readable_status(sensor, setting):
        doors = ["door", "cupboard", "freezer", "fridge", "dishwasher", "washingmachine"]
        def has_door():
            for d in doors:
                if d in sensor.lower():
                   return True
            return False

        if has_door():
            return ("Closed", "Open")[setting]
        else:
            return ("Off", "On")[setting]

    #replace each sensor occurrence with one data item for "on"/"open" and one for "off"/"close"
    def status_change_iterator(sensor_data):
        for start, end, sensor in sensor_data:
            yield start, sensor, human_readable_status(sensor, 1)
            yield end, sensor, human_readable_status(sensor, 0)

    #actual formatting happens here
    formatted_data = map(initial_formatting,raw_sensor_data)
    formatted_data = remove_unknown_sensors(formatted_data)
    formatted_data = [d for d in status_change_iterator(formatted_data)]

    return formatted_data


def write_sensor_data_csv(house, sensor_data):
    data=DataFrame(sensor_data, columns=["timestamp", "sensor", "value"])
    data=data.set_index("timestamp")
    data=data.sort()
    data.to_csv("%s.csv"%house)


def convert_house(matlab_file,house):

    #read data and perform initial extraction of raw data
    matlab_data = loadmat(matlab_file)[house]
    raw_data = extract_raw_data(matlab_data)

    #clean up sensor and activity names
    replace_whitespace = lambda n: n.replace(" ", "_")
    sensors = {sid: replace_whitespace(name) for sid, name in raw_data["sensors"].items()}
    #activities = {aid: replace_whitespace(name) for aid, name in raw_data["activities"].items()}

    #format the raw sensor data to desired format and write to file
    sensor_data = format_sensor_data(raw_data["sensor_data"], sensors)
    write_sensor_data_csv(house, sensor_data)


#location of the matlab file is given by command line argument
parser = argparse.ArgumentParser(description='Convert the Kasteren datasets into CSV files')
parser.add_argument("datafile", help="matlab file that contains the datasets")
args = parser.parse_args()

#convert houseA and houseB datasets to CSV files
convert_house(args.datafile, "houseA")
convert_house(args.datafile, "houseB")

