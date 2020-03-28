import sys
import tensorflow as tf


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders"""
    features = {
        'f1': tf.placeholder(dtype=tf.float32, shape=[None, 1]),
        'f2': tf.placeholder(dtype=tf.float32, shape=[None, 1]),
        'f3': tf.placeholder(dtype=tf.float32, shape=[None, 1]),
        'f4': tf.placeholder(dtype=tf.float32, shape=[None, 1])
    }
    receiver_tensors = {
        'f5': tf.placeholder(dtype=tf.int32, shape=[None, 1])
    }
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def run(model_dir, output_dir):
    # Define feature columns number
    feature_types = ['f1', 'f2', 'f3', 'f4']
    feature_columns = [tf.feature_column.numeric_column(key=key) for key in feature_types]

    # Define classifier.
    classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=3,
        model_dir=model_dir
    )

    classifier.export_saved_model(
        export_dir_base=output_dir,
        serving_input_receiver_fn=serving_input_receiver_fn
    )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
