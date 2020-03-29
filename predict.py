import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Directory of the trained model.', default='models', type=str)
    parser.add_argument('--inputs',
                        help='Array of inputs.',
                        default='6.4,2.9,4.3,1.3',
                        type=lambda s: [float(item) for item in s.split(',')])

    args = parser.parse_args()
    assert (len(args.inputs) == 4), "Inputs can only consist of 4 items."

    run(args.model_dir, args.inputs)
