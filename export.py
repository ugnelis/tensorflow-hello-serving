import argparse
import tensorflow as tf


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders"""
    feature_placeholders = {
        'f1': tf.placeholder(dtype=tf.float32, shape=[None]),
        'f2': tf.placeholder(dtype=tf.float32, shape=[None]),
        'f3': tf.placeholder(dtype=tf.float32, shape=[None]),
        'f4': tf.placeholder(dtype=tf.float32, shape=[None])
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=feature_placeholders
    )


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Directory of the trained model.', default='models', type=str)
    parser.add_argument('--output_dir',
                        help='Directory of output protobuf (.pb) model.',
                        default='exported_models/hello',
                        type=str)

    args = parser.parse_args()
    run(args.model_dir, args.output_dir)
