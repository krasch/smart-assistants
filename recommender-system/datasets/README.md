Using smart home datasets with the recommender system
======

Required data format
---------------------

The recommender system expects data in a comma-seperated value file (CSV) with following columns:

    timestamp, sensor, value

An example (randomly-generated, useless) dataset can be found in file _example.csv_. The [pandas data analysis library](http://pandas.pydata.org/) supports several time formats, please see the [documentation](http://pandas.pydata.org/pandas-docs/dev/io.html#date-parsing-functions) for more details.


Using the houseA and houseB datasets by T.L.M. van Kasteren et al.
----------------------

For evaluation of the recommender system we used the houseA and houseB datasets that are described in:

* Accurate Activity Recognition in a Home Setting. T.L.M. van Kasteren, A. K. Noulas, G. Englebienne and B.J.A. Kröse, in Proceedings of ACM Tenth International Conference on Ubiquitous Computing (Ubicomp'08). Seoul, South Korea, 2008.

* Transferring Knowledge of Activity Recognition across Sensor Networks, T.L.M. van Kasteren, G.    Englebienne and B.J.A. Kröse, In Proceedings of the Eighth International Conference on Pervasive Computing (Pervasive 2010). Helsinki, Finland, 2010. 

To convert these datasets to a CSV file in the required format:

1. Download the zip archive from https://sites.google.com/site/tim0306/tlDatasets.zip
2. Open zip archive and extract file tlDatasets/tlDatasets.mat to the current directory 
3. Run `python converters/matlab.py tlDatasets.mat`

This will create two files "houseA.csv" and "houseB.csv".
