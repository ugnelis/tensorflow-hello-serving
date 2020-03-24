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

# In my current Pandas version, it didn't detect integer and float classes,
# that is why I forced conversion while reading form file with convert_numeric=True
# (all columns became float, but actually f5 is integer, so I reconverted it to int.

# Three  output classes setosa=0, versicolor=1, virginica=2.


# Read data from csv file.
train_data = pd.read_csv("dataset/iris_training.csv", names=['f1', 'f2', 'f3', 'f4', 'f5'])\
    .convert_objects(convert_numeric=True)
test_data = pd.read_csv("dataset/iris_test.csv", names=['f1', 'f2', 'f3', 'f4', 'f5'])\
    .convert_objects(convert_numeric=True)

# Remove headers (the first row's).
train_data = train_data.drop(0)
test_data = test_data.drop(0)

# Separate train data.
train_x = train_data[['f1', 'f2', 'f3', 'f4']]
train_y = train_data.loc[:, ['f5']].astype(int)

# Separate test data.
test_x = test_data[['f1', 'f2', 'f3', 'f4']]
test_y = test_data.loc[:, ['f5']].astype(int)

# Define feature columns
feature_columns = [tf.feature_column.numeric_column(key=key) for key in train_x.keys()]

# Define classifier
classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=3)

classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, 100),
    steps=2000)


eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, 100))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
