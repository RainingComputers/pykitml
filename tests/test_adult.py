from pykitml.testing import pktest_graph


@pktest_graph
def test_adult():
    import os.path

    import pykitml as pk
    from pykitml.datasets import adult

    # Download the dataset
    if not os.path.exists('adult.data.pkl'):
        adult.get()

    # Load adult data set
    inputs_train, outputs_train, inputs_test, outputs_test = adult.load()

    # Normalize dataset
    array_min, array_max = pk.get_minmax(inputs_train)
    inputs_train = pk.normalize_minmax(inputs_train, array_min, array_max, cols=[0, 2, 9, 10, 11])
    inputs_test = pk.normalize_minmax(inputs_test, array_min, array_max, cols=[0, 2, 9, 10, 11])

    # Convert categorical values to one-hot values
    inputs_train, inputs_test = pk.onehot_cols_traintest(inputs_train, inputs_test, cols=[1, 3, 4, 5, 6, 7, 8, 9, 12])

    # Create model
    adult_classifier = pk.LogisticRegression(104, 1)

    # Train the model
    adult_classifier.train(
        training_data=inputs_train,
        targets=outputs_train,
        batch_size=10,
        epochs=1500,
        optimizer=pk.Adam(learning_rate=0.015, decay_rate=0.99),
        testing_data=inputs_test,
        testing_targets=outputs_test,
        testing_freq=30,
        decay_freq=40
    )

    # Save it
    pk.save(adult_classifier, 'adult_classifier.pkl')

    # Plot performance
    adult_classifier.plot_performance()

    # Print accuracy
    accuracy = adult_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = adult_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    adult_classifier.confusion_matrix(inputs_test, outputs_test)

    # Assert if it has enough accuracy
    assert adult_classifier.accuracy(inputs_test, outputs_test) >= 82


if __name__ == '__main__':
    try:
        test_adult.__wrapped__()
    except AssertionError:
        pass
