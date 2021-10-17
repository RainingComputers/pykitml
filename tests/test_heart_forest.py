from pykitml.testing import pktest_graph, pktest_nograph


@pktest_graph
def test_heart_forest():
    import os.path

    import pykitml as pk
    from pykitml.datasets import heartdisease

    # Download the dataset
    if not os.path.exists('heartdisease.pkl'):
        heartdisease.get()

    # Load heart data set
    inputs, outputs = heartdisease.load()
    outputs = pk.onehot(outputs)

    # Create model
    ftypes = [
        'continues', 'categorical', 'categorical',
        'continues', 'continues', 'categorical', 'categorical',
        'continues', 'categorical', 'continues', 'categorical',
        'categorical', 'categorical'
    ]
    forest_heart_classifier = pk.RandomForest(13, 2, max_depth=8, feature_type=ftypes)

    # Train
    forest_heart_classifier.train(inputs, outputs)

    # Save it
    pk.save(forest_heart_classifier, 'forest_heart_classifier.pkl')

    # Print accuracy
    accuracy = forest_heart_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    forest_heart_classifier.confusion_matrix(inputs, outputs,
                                             gnames=['False', 'True'])

    # Assert accuracy
    assert (forest_heart_classifier.accuracy(inputs, outputs)) >= 94


@pktest_nograph
def test_predict_heart_forest():
    import numpy as np
    import pykitml as pk

    # Predict heartdisease for a person with
    # age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal
    # 67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3
    input_data = np.array([67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3], dtype=float)

    # Load the model
    forest_heart_classifier = pk.load('forest_heart_classifier.pkl')

    # Get output
    forest_heart_classifier.feed(input_data)
    model_output = forest_heart_classifier.get_output()

    # Print result (log of probabilities)
    print(model_output)


if __name__ == '__main__':
    try:
        test_heart_forest.__wrapped__()
        test_predict_heart_forest.__wrapped__()
    except AssertionError:
        pass
