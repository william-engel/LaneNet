import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('INFO')

class MeanIoU_FromLogits(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name=None, **kwargs):
        super(MeanIoU_FromLogits, self).__init__(name=name, **kwargs)
        self.MeanIoU = tf.keras.metrics.MeanIoU(num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.softmax(y_pred)
        y_pred = tf.argmax(y_pred, axis = -1)

        y_true = tf.argmax(y_true, axis = -1)

        return self.MeanIoU.update_state(y_true, y_pred)

    def result(self):
        return self.MeanIoU.result()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.MeanIoU.reset_states()