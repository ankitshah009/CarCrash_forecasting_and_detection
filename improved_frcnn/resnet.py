from keras.layers import TimeDistributed, Dense, Flatten
from improved_frcnn.layers import ContextualROIPoolConvLayer
from analysis.keras_frcnn.resnet import classifier_layers


def classifier(step_stride, n_contexts, base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)

    out_roi_pool = ContextualROIPoolConvLayer(step_stride=step_stride, n_contexts=n_contexts,
                                              pool_size=pooling_regions, num_rois=num_rois)([base_layers, input_rois])

    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]
