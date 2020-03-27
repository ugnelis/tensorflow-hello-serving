import sys
import tensorflow as tf


def predict_input_fn(features):
    dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    return dataset.batch(1)


def run(model_dir, inputs):
    # Define feature columns number
    feature_types = ['f1', 'f2', 'f3', 'f4']
    feature_columns = [tf.feature_column.numeric_column(key=key) for key in feature_types]

    # Define classifier.
    classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=3,
        model_dir=model_dir
    )

    # Define prediction dictionary. Note: the every 'f' input should be as array
    # because Dataset formation is made with tf.data.Dataset.from_tensor_slices().
    predict_x = {
        'f1': [inputs[0]],
        'f2': [inputs[1]],
        'f3': [inputs[2]],
        'f4': [inputs[3]],
    }

    # Predict.
    prediction_result = classifier.predict(
        input_fn=lambda: predict_input_fn(predict_x))

    for result in prediction_result:
        probabilities = result['probabilities'] * 100
        print('setosa: {0:0.3f}%'.format(probabilities[0]))
        print('versicolor: {0:0.3f}%'.format(probabilities[1]))
        print('virginica: {0:0.3f}%'.format(probabilities[2]))


if __name__ == "__main__":
    run(sys.argv[1], [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])])
