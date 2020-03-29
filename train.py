import argparse
import pandas as pd
import tensorflow as tf


def read_data(path):
    # Read data from csv file and remove the first row.
    # Three output classes setosa=0, versicolor=1, virginica=2.
    data = pd.read_csv(path, skiprows=1, names=['f1', 'f2', 'f3', 'f4', 'f5'])
    x = data[['f1', 'f2', 'f3', 'f4']]
    y = data.loc[:, ['f5']]
    return x, y


def train_input_fn(features, labels, batch_size):
    inputs = (dict(features), labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    inputs = (dict(features), labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    return dataset.batch(batch_size)


def run(train_set_path, test_set_path, model_dir):
    # Read train data.
    train_x, train_y = read_data(train_set_path)

    # Read test data.
    test_x, test_y = read_data(test_set_path)

    # Define feature columns.
    feature_columns = [tf.feature_column.numeric_column(key=key) for key in train_x.keys()]

    # Define classifier.
    classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=3,
        model_dir=model_dir
    )

    # Run training.
    classifier.train(
        input_fn=lambda: train_input_fn(train_x, train_y, 100),
        steps=2000)

    # # Run testing.
    # eval_result = classifier.evaluate(
    #     input_fn=lambda: eval_input_fn(test_x, test_y, 100))

    # print('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_path',
                        help='Path to a .csv of the train data set.',
                        default='dataset/iris_training.csv',
                        type=str)
    parser.add_argument('--test_set_path',
                        help='Path to a .csv of the test data set.',
                        default='dataset/iris_test.csv',
                        type=str)
    parser.add_argument('--model_dir', help='Directory of the model.', default='models', type=str)

    args = parser.parse_args()

    run(args.train_set_path, args.test_set_path, args.model_dir)
