from pykitml.testing import pktest_graph


@pktest_graph
def test_pca_compression():
    import os.path
    import random

    import matplotlib.pyplot as plt
    import pykitml as pk
    from pykitml.datasets import mnist

    # Download dataset
    if not os.path.exists('mnist.pkl'):
        mnist.get()

    # Load dataset
    training_data, _, _, _ = mnist.load()

    # Train PCA, reduce 784 dimensions to 250 dimensions
    pca = pk.PCA(training_data, 250)
    print('Variance retention:', pca.retention)

    # Pick random datapoints
    indices = random.sample(range(1, 1000), 16)
    examples = training_data[indices]

    # Show the original images
    plt.figure('Original', figsize=(10, 7))
    for i in range(1, 17):
        plt.subplot(4, 4, i)
        plt.imshow(examples[i-1].reshape((28, 28)), cmap='gray')

    # Transform the example and compress
    transformed_examples = pca.transform(examples)

    # Inverse transform and recover the examples
    recovered_examples = pca.inverse_transform(transformed_examples)

    # Show the inverse transformed examples
    plt.figure('Recovered', figsize=(10, 7))
    for i in range(1, 17):
        plt.subplot(4, 4, i)
        plt.imshow(recovered_examples[i-1].reshape((28, 28)), cmap='gray')

    # Show results
    plt.show()


if __name__ == '__main__':
    test_pca_compression.__wrapped__()
