## Introduction

This repository contains some of the code I wrote during my PhD studies. In particular it contains the implementation
of a recommender system for smart homes that aims to improve smart home usability through personalisation. To do that,
the recommender system continuously interprets the user's current situation and recommends services
that fit the user's habits, i.e. automate some action that the user would want to perform anyway. 

More information can be found here:

* Katharina Rasch. [An unsupervised recommender system for smart homes](http://iospress.metapress.com/content/372n5686n0426558/),
Journal of Ambient Intelligence and Smart Environments, 6 (1): 21-73, 2014
* Katharina Rasch. [Smart assistants for smart homes](http://www.diva-portal.org/smash/get/diva2:650328/FULLTEXT01.pdf).
PhD thesis, KTH Royal Institute of Technology, Stockholm, Sweden, 2014

(You will probably be a bit lost here, if you have not read at least the first paper, unfortunately.)

## Prerequisites

The recommender system is written in Python 2.7. All necessary libraries are listed in file [requirements.txt](requirements.txt)
and can be installed by running:

    pip install -r requirements.txt

## Smart home datasets

You will also need some smart home dataset that the recommender system can be run on. To get started, it will be easiest
to download and convert the smart home datasets collected by van Kasteren et al. Please see the Readme file in
the [datasets](datasets) directory for instructions.


## Quick usage

To compare the proposed system with a Naive Bayes classifier using the Kasteren "houseA" dataset, run:

    cd examples
    python compare_classifiers.py

This should (after a short while) generate output similar to the following:

                      Recall        Precision               F1
    Our method   0.60 +/- 0.0179  0.65 +/- 0.0328  0.58 +/- 0.0196
    Naive Bayes  0.48 +/- 0.0343  0.38 +/- 0.0252  0.40 +/- 0.0303
    Random       0.07 +/- 0.0072  0.27 +/- 0.0307  0.09 +/- 0.0104

    [3 rows x 3 columns]

                    Training time    Overall testing time   Individual testing time
    Our method   765.32 +/- 17.0273   298.40 +/- 10.4052         1.29 +/- 0.0453
    Naive Bayes    30.15 +/- 1.1927     48.97 +/- 1.7049         0.21 +/- 0.0073
    Random          0.03 +/- 0.0026     28.71 +/- 1.9472         0.12 +/- 0.0084

    [3 rows x 3 columns]

The exact results in the second table printed depend on the performance of your computer. The results in the first table
should be very similar to those listed above.

The system will also generate some plots that compare the classifiers graphically; you can find those in the "plots/houseA"
directory.

All generated results are explained in detail in the accompanying paper
["An unsupervised recommender system for smart homes"](http://iospress.metapress.com/content/372n5686n0426558/).

## Further exploration

##### Want to run all the other evaluation experiments described in the paper?
-------------

Scripts for all experiments can be found in the [examples](examples) directory. The examples directory also contains additional evaluations not described in the paper, e.g. for printing a confusion matrix. 


##### Want to used your own data?
------------------------------

The necessary data format is described in the Readme file in the [datasets](datasets) directory. Make sure to also
 include your dataset in script you want to run in the [examples](examples) directory.

##### Want to see the core sourcecode for the proposed recommender system?
------------------------------

You can find it in [recsys/classifiers/temporal.py](recsys/classifiers/temporal.py).


