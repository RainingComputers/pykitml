import cProfile
import os.path

import numpy as np
import pykitml as pk
from pykitml import heartdisease

def test_download():
    heartdisease.get()

def test_heart_bayes():
    # Load heart data set
    inputs, outputs = heartdisease.load()
    outputs = pk.onehot(outputs)

    distrbutions = [
        'normal', 'binomial', 'multinomial',
        'normal', 'normal', 'binomial', 'multinomial',
        'normal', 'binomial', 'normal', 'multinomial',
        'multinomial', 'multinomial'
    ]

    # Create model
    bayes_heart_classifier = pk.NaiveBayes(13, 2, distrbutions)

    # Train
    bayes_heart_classifier.train(inputs, outputs)

    # Check accuracy
    assert (bayes_heart_classifier.accuracy(inputs, outputs)) > 73

    # Save it
    pk.save(bayes_heart_classifier, 'bayes_heart_classifier.pkl')

if __name__ == '__main__':
    # Download the dataset 
    if(not os.path.exists('heartdisease.pkl')): heartdisease.get()
    
    # Train
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_heart_bayes)
        profiler.dump_stats('test_heart_bayes.dat') 
    except AssertionError:
        pass

    # Load dataset
    inputs, outputs = heartdisease.load()
    outputs = pk.onehot(outputs)
    
    # Load model
    bayes_heart_classifier = pk.load('bayes_heart_classifier.pkl')

    # Print accuracy
    accuracy = bayes_heart_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    bayes_heart_classifier.confusion_matrix(inputs, outputs, 
        gnames=['False', 'True'])