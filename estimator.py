import sys
import pandas as pd
import tensorflow as tf


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    return dataset.batch(batch_size)


def run(train_set_path, test_set_path):
    # Read data from csv file and remove the first row.
    # Three  output classes setosa=0, versicolor=1, virginica=2.
    train_data = pd.read_csv(train_set_path, skiprows=1, names=['f1', 'f2', 'f3', 'f4', 'f5'])
    test_data = pd.read_csv(test_set_path, skiprows=1, names=['f1', 'f2', 'f3', 'f4', 'f5'])

    # Separate train data.
    train_x = train_data[['f1', 'f2', 'f3', 'f4']]
    train_y = train_data.loc[:, ['f5']]

    # Separate test data.
    test_x = test_data[['f1', 'f2', 'f3', 'f4']]
    test_y = test_data.loc[:, ['f5']]

    # Define feature columns
    feature_columns = [tf.feature_column.numeric_column(key=key) for key in train_x.keys()]

    # Define classifier
    classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=3)

    # Run training.
    classifier.train(
        input_fn=lambda: train_input_fn(train_x, train_y, 100),
        steps=2000)

    # Run testing.
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_x, test_y, 100))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
