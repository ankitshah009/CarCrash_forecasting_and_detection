from analysis.keras_frcnn.roi_pooling_conv import RoiPoolingConv
import keras.backend as K
import tensorflow as tf


class ContextualROIPoolConvLayer(RoiPoolingConv):
    def __init__(self, step_stride=2, n_contexts=1, **kwargs):
        self.step_stride = step_stride
        self.n_contexts = n_contexts
        super(ContextualROIPoolConvLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ContextualROIPoolConvLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        if self.dim_ordering == 'th':
            raise NotImplementedError("We don't use Theano backend.")

        outputs = []
        s = K.shape(img)

        for roi_idx in range(self.num_rois):
            contexts = []
            for i in range(self.n_contexts):
                stride = i * self.step_stride

                x = rois[0, roi_idx, 0] - stride
                y = rois[0, roi_idx, 1] - stride
                w = rois[0, roi_idx, 2] + 2 * stride
                h = rois[0, roi_idx, 3] + 2 * stride
                x = K.cast(x, 'int32')
                if (x < 0) is not None: x = 0
                y = K.cast(y, 'int32')
                if (y < 0) is not None: y = 0
                w = K.cast(w, 'int32')
                if (w > s[0]) is not None: w = s[0]
                h = K.cast(h, 'int32')
                if (h > s[1]) is not None: h = s[1]

                rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
                contexts.append(rs)
            # Apply maxout
            C = K.concatenate(contexts, axis=0)
            T = K.max(C, axis=0)
            outputs.append(T)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output


class AugmentedContextualROIPoolConvLayer(ContextualROIPoolConvLayer):
    def __init__(self, **kwargs):
        super(AugmentedContextualROIPoolConvLayer, self).__init__(**kwargs)
