from pykitml.testing import pktest_nograph


@pktest_nograph
def test_boston_tree():
    import pykitml as pk
    from pykitml.datasets import boston

    import os

    # Download the dataset
    if not os.path.exists('boston.pkl'):
        boston.get()

    # Load heart data set
    inputs_train, outputs_train, inputs_test, outputs_test = boston.load()

    # Create model
    ftypes = [
        'continues', 'continues', 'continues',
        'categorical', 'continues', 'continues',
        'continues', 'continues', 'continues',
        'continues', 'continues', 'continues', 'continues'
    ]
    tree_boston = pk.DecisionTree(13, 1, feature_type=ftypes, max_depth=8, min_split=20, regression=True)

    # Train
    tree_boston.train(inputs_train, outputs_train)

    # Print r2score
    r2score_train = tree_boston.r2score(inputs_train, outputs_train)
    print('Train r2score:', r2score_train)
    r2score = tree_boston.r2score(inputs_test, outputs_test)
    print('Test r2score:', r2score)

    # Show the tree
    tree_boston.show_tree()

    # Assert r2score
    assert r2score_train > 0.9


if __name__ == '__main__':
    try:
        test_boston_tree.__wrapped__()
    except AssertionError:
        pass
