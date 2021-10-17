from pykitml.testing import pktest_graph


@pktest_graph
def test_s1_kmeans():
    import os

    import pykitml as pk
    from pykitml.datasets import s1clustering
    import matplotlib.pyplot as plt

    # Download the dataset
    if not os.path.exists('s1.pkl'):
        s1clustering.get()

    # Load the dataset
    train_data = s1clustering.load()

    # Run KMeans
    clusters, cost = pk.kmeans(train_data, 15)

    # Plot dataset, x and y
    plt.scatter(train_data[:, 0], train_data[:, 1])

    # Plot clusters, x and y
    plt.scatter(clusters[:, 0], clusters[:, 1], c='red')

    # Show graph
    plt.show()

    # Assert cost
    assert cost <= 1790000000


if __name__ == '__main__':
    try:
        test_s1_kmeans.__wrapped__()
    except AssertionError:
        pass
