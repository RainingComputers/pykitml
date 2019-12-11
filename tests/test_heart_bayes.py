import cProfile
import os.path

import numpy as np
import pykitml as pk
from pykitml import heartdisease

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_download():
    heartdisease.get()

def test_heart_bayes():
    import os.path

    import numpy as np
    import pykitml as pk
    from pykitml import heartdisease

    # Download the dataset 
    if(not os.path.exists('heartdisease.pkl')): heartdisease.get()

    # Load heart data set
    inputs, outputs = heartdisease.load()

    # Change 0/False to [1, 0]
    # Change 1/True to [0, 1]
    outputs = pk.onehot(outputs)

    distrbutions = [
        'gaussian', 'binomial', 'multinomial',
        'gaussian', 'gaussian', 'binomial', 'multinomial',
        'gaussian', 'binomial', 'gaussian', 'multinomial',
        'multinomial', 'multinomial'
    ]

    # Create model
    bayes_heart_classifier = pk.NaiveBayes(13, 2, distrbutions)

    # Train
    bayes_heart_classifier.train(inputs, outputs)

    # Save it
    pk.save(bayes_heart_classifier, 'bayes_heart_classifier.pkl')

    # Print accuracy
    accuracy = bayes_heart_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    bayes_heart_classifier.confusion_matrix(inputs, outputs, 
        gnames=['False', 'True'])

    # Assert accuracy
    assert (bayes_heart_classifier.accuracy(inputs, outputs)) > 84

if __name__ == '__main__':
    # Train
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_heart_bayes)
        profiler.dump_stats('test_heart_bayes.dat') 
    except AssertionError:
        pass