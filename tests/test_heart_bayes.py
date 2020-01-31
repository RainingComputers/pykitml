import cProfile
import os.path

import numpy as np
import pykitml as pk
from pykitml.datasets import heartdisease

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_download():
    heartdisease.get()

def test_heart_bayes():
    import os.path

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import heartdisease

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

def test_predict():
    import os.path

    import numpy as np
    import pykitml as pk

    # Predict heartdisease for a person with
    # age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal
    # 67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3 
    input_data = np.array([67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3], dtype=float)

    # Load the model
    bayes_heart_classifier = pk.load('bayes_heart_classifier.pkl')

    # Get output
    bayes_heart_classifier.feed(input_data)
    model_output = bayes_heart_classifier.get_output()

    # Print result (log of probabilities)
    print(model_output)

if __name__ == '__main__':
    # Train
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_heart_bayes)
        profiler.dump_stats('test_heart_bayes.dat') 

        test_predict()
    except AssertionError:
        pass