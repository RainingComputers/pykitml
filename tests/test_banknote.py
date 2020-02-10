from pykitml.testing import pktest_graph, pktest_nograph

@pktest_graph
def test_banknote():
    import os.path

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import banknote

    # Download the dataset 
    if(not os.path.exists('banknote.pkl')): banknote.get()

    # Load banknote data set
    inputs_train, outputs_train, inputs_test, outputs_test = banknote.load()

    # Normalize dataset
    array_min, array_max = pk.get_minmax(inputs_train)
    inputs_train = pk.normalize_minmax(inputs_train, array_min, array_max)
    inputs_test = pk.normalize_minmax(inputs_test, array_min, array_max)

    # Create polynomial features
    inputs_train_poly = pk.polynomial(inputs_train)
    inputs_test_poly = pk.polynomial(inputs_test)

    # Create model
    banknote_classifier = pk.LogisticRegression(inputs_train_poly.shape[1], 1)

    # Train the model
    banknote_classifier.train(
        training_data=inputs_train_poly,
        targets=outputs_train, 
        batch_size=10, 
        epochs=1500, 
        optimizer=pk.Adam(learning_rate=0.06, decay_rate=0.99),
        testing_data=inputs_test_poly,
        testing_targets=outputs_test, 
        testing_freq=30,
        decay_freq=40
    )

    # Save it
    pk.save(banknote_classifier, 'banknote_classifier.pkl') 

    # Plot performance
    banknote_classifier.plot_performance()
    
    # Print accuracy
    accuracy = banknote_classifier.accuracy(inputs_train_poly, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = banknote_classifier.accuracy(inputs_test_poly, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    banknote_classifier.confusion_matrix(inputs_test_poly, outputs_test)

    # Assert if it has enough accuracy
    assert banknote_classifier.accuracy(inputs_test_poly, outputs_test) >= 99

@pktest_nograph
def test_predict_banknote():
    import os.path

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import banknote

    # Predict banknote validity with variance, skewness, curtosis, entropy
    # of -2.3, -9.3, 9.37, -0.86

    # Load banknote data set
    inputs_train, outputs_train, inputs_test, outputs_test = banknote.load()

    # Load the model
    banknote_classifier = pk.load('banknote_classifier.pkl')

    # Normalize the inputs
    array_min, array_max = pk.get_minmax(inputs_train)
    input_data = pk.normalize_minmax(np.array([-2.3, -9.3, 9.37, -0.86]), array_min, array_max)

    # Create polynomial features
    input_data_poly = pk.polynomial(input_data)

    # Get output
    banknote_classifier.feed(input_data_poly)
    model_output = banknote_classifier.get_output()

    # Print result
    print(model_output)  

if __name__ == '__main__':
    try:
        test_banknote.__wrapped__()
        test_predict_banknote.__wrapped__()
    except AssertionError:
        pass