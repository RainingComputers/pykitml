from pykitml.testing import pktest_graph, pktest_nograph

@pktest_nograph
def test_boston_forest():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import boston

    import os

    # Download the dataset
    if(not os.path.exists('boston.pkl')): boston.get()

    # Load heart data set
    inputs_train, outputs_train, inputs_test, outputs_test = boston.load()

    # Create model
    ftypes = [
        'continues', 'continues', 'continues',
        'categorical', 'continues', 'continues',
        'continues', 'continues', 'continues', 
        'continues', 'continues', 'continues', 'continues'
    ]
    forest_boston = pk.RandomForest(13, 1, feature_type=ftypes, max_depth=4, min_split=20, regression=True)

    # Train
    forest_boston.train(inputs_train, outputs_train)

    # Print r2score
    r2score_train = forest_boston.r2score(inputs_train, outputs_train)
    print('Train r2score:', r2score_train)
    r2score = forest_boston.r2score(inputs_test, outputs_test)
    print('Test r2score:', r2score)

    # Assert r2score
    assert r2score_train > 0.7

if __name__ == '__main__':
    try:
        test_boston_forest.__wrapped__()
    except AssertionError:
        pass