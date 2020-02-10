from pykitml.testing import pktest_graph, pktest_nograph

@pktest_graph
def test_banknote_forest():
    import os

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import banknote

    # Download the dataset 
    if(not os.path.exists('banknote.pkl')): banknote.get()

    # Load heart data set
    inputs_train, outputs_train, inputs_test, outputs_test = banknote.load()
    
    # Change 0/False to [1, 0]
    # Change 1/True to [0, 1]
    outputs_train = pk.onehot(outputs_train)
    outputs_test = pk.onehot(outputs_test)

    # Create model
    ftypes = ['continues']*4
    forest_banknote_classifier = pk.RandomForest(4, 2, max_depth=9, feature_type=ftypes)

    # Train
    forest_banknote_classifier.train(inputs_train, outputs_train)

    # Save it
    pk.save(forest_banknote_classifier, 'forest_banknote_classifier.pkl')

    # Print accuracy
    accuracy = forest_banknote_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = forest_banknote_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    forest_banknote_classifier.confusion_matrix(inputs_test, outputs_test, 
        gnames=['False', 'True'])

    # Assert accuracy
    assert (forest_banknote_classifier.accuracy(inputs_test, outputs_test)) >= 98

@pktest_nograph
def test_predict_banknote_forest():
    import os.path

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import banknote

    # Predict banknote validity with variance, skewness, curtosis, entropy
    # of -2.3, -9.3, 9.37, -0.86
    input_data = np.array([-2.3, -9.3, 9.37, -0.86])

    # Load the model
    forest_banknote_classifier = pk.load('forest_banknote_classifier.pkl')

    # Get output
    forest_banknote_classifier.feed(input_data)
    model_output = forest_banknote_classifier.get_output()

    # Print result
    print(model_output)  

if __name__ == '__main__':
    try:
        test_banknote_forest.__wrapped__()
        test_predict_banknote_forest.__wrapped__()
    except AssertionError:
        pass

