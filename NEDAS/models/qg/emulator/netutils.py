import os
import netCDF4
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class Att_Res_UNet():
    def __init__(self, list_predictors, list_targets, patch_dim, batch_size, n_filters, activation, kernel_initializer, batch_norm, pooling_type, dropout):
        self.list_predictors = list_predictors
        self.list_targets = list_targets
        self.patch_dim = patch_dim
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm = batch_norm
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.n_predictors = len(list_predictors)
        self.n_targets = len(list_targets)

    def repeat_elem(self, tensor, rep):
        return tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis = 3), arguments = {'repnum': rep})(tensor)

    def gating_signal(self, x, n_filters, batch_norm = False):
        x = tf.keras.layers.Conv2D(n_filters, (1,1), padding = "same")(x)
        if batch_norm == True:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return(x)

    def attention_block(self, x, g, inter_shape):
        shape_x = tf.keras.backend.int_shape(x)
        shape_g = tf.keras.backend.int_shape(g)

        theta_x = tf.keras.layers.Conv2D(inter_shape, kernel_size = (2,2), strides = (2,2), padding = "same")(x)
        shape_theta_x = tf.keras.backend.int_shape(theta_x)

        phi_g = tf.keras.layers.Conv2D(inter_shape, kernel_size = (1,1), padding = "same")(g)
        upsample_g = tf.keras.layers.Conv2DTranspose(inter_shape, (3,3), 
                                                     strides = (shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                                     padding = "same")(phi_g)

        concat_xg = tf.keras.layers.add([upsample_g, theta_x])
        act_xg = tf.keras.layers.Activation("relu")(concat_xg)

        psi = tf.keras.layers.Conv2D(1, (1,1), padding = "same")(act_xg)
        sigmoid_xg = tf.keras.layers.Activation("sigmoid")(psi)
        shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)

        upsample_psi = tf.keras.layers.UpSampling2D(size = (shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
        upsample_psi = self.repeat_elem(upsample_psi, shape_x[3])
        y = tf.keras.layers.multiply([upsample_psi, x])

        result = tf.keras.layers.Conv2D(shape_x[3], (1,1), padding = "same")(y)
        result_bn = tf.keras.layers.BatchNormalization()(result)

        return(result_bn)

    def residual_conv_block(self, x, n_filters, padding = "same", kernel_size=(3,3)):
        conv = tf.keras.layers.Conv2D(n_filters, kernel_size = kernel_size, padding = padding, kernel_initializer = self.kernel_initializer)(x)
        if self.batch_norm == True:
            conv = tf.keras.layers.BatchNormalization(axis = 3)(conv)
        conv = tf.keras.layers.Activation(self.activation)(conv)

        conv = tf.keras.layers.Conv2D(n_filters, kernel_size = kernel_size, padding = padding, kernel_initializer = self.kernel_initializer)(conv)
        if self.batch_norm == True:
            conv = tf.keras.layers.BatchNormalization(axis = 3)(conv)

        shortcut = tf.keras.layers.Conv2D(n_filters, kernel_size = (1,1), padding = padding)(x)
        if self.batch_norm == True:
            shortcut = tf.keras.layers.BatchNormalization(axis = 3)(shortcut)

        res_path = tf.keras.layers.add([shortcut, conv])
        res_path = tf.keras.layers.Activation(self.activation)(res_path)

        return(res_path)

    def downsample_block(self, x, n_filters, pool_size = (2,2), kernel_size = (3,3), strides = 2):
        f = self.residual_conv_block(x, n_filters, kernel_size=kernel_size)

        if self.pooling_type == "Max":
            p = tf.keras.layers.MaxPool2D(pool_size = pool_size, strides = strides)(f)
        elif self.pooling_type == "Average":
            p = tf.keras.layers.AveragePooling2D(pool_size = pool_size, strides = strides)(f)

        p = tf.keras.layers.Dropout(self.dropout)(p)
        return(f, p)  

    def upsample_block(self, x, conv_features, n_filters, kernel_size = (3,3), strides = 2, padding = "same"):
        gating = self.gating_signal(x, n_filters)
        att = self.attention_block(conv_features, gating, n_filters)
        up_att = tf.keras.layers.UpSampling2D(size = (2, 2), data_format = "channels_last")(x)
        up_att = tf.keras.layers.concatenate([up_att, att], axis = 3)
        up_conv = self.residual_conv_block(up_att, n_filters,kernel_size=kernel_size)
        return(up_conv)

    def make_unet_model(self): 
        inputs = tf.keras.layers.Input(shape = (*self.patch_dim, self.n_predictors))
        # Encoder (downsample)
        f1, p1 = self.downsample_block(inputs, self.n_filters[0], kernel_size=(7,7))
        f2, p2 = self.downsample_block(p1, self.n_filters[1])
        f3, p3 = self.downsample_block(p2, self.n_filters[2])
        f4, p4 = self.downsample_block(p3, self.n_filters[3])
        f5, p5 = self.downsample_block(p4, self.n_filters[4])
        # Bottleneck
        u5 = self.residual_conv_block(p5, self.n_filters[5])
        # Decoder (upsample)
        u4 = self.upsample_block(u5, f5, self.n_filters[4])
        u3 = self.upsample_block(u4, f4, self.n_filters[3])
        u2 = self.upsample_block(u3, f3, self.n_filters[2])
        u1 = self.upsample_block(u2, f2, self.n_filters[1])
        u0 = self.upsample_block(u1, f1, self.n_filters[0])
        # outputs
        SICerror = tf.keras.layers.Conv2D(self.n_targets, (1, 1), padding = "same", activation = "linear", dtype = tf.float32, name = "psi")(u0)
        unet_model = tf.keras.Model(inputs, SICerror, name = "U-Net")

        return(unet_model)

    def featname2tuple(feature_name):
        parts = feature_name.rsplit('_', 1)  # Split from the right, max 1 split
        varname = parts[0]
        channel = int(parts[1])
        return varname, channel

class Data_generator(tf.keras.utils.Sequence):
    def __init__(self, nrun, startrun, shuffle, batch_size, dim,  path_data, list_predictors, list_targets, sampleperrun = 100):
        self.nrun = nrun
        self.startrun = startrun
        self.sampleperrun = sampleperrun
        self.n = nrun*(sampleperrun-1)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.path_data = path_data
        self.dim = dim
        self.shuffle = shuffle
        self.list_predictors = list_predictors
        self.list_targets = list_targets
        self.npredictors = len(self.list_predictors)
        self.ntargets = len(self.list_targets)
        self.indexes = np.arange(self.n)
        if self.shuffle == True:
            rng = np.random.default_rng()
            rng.shuffle(self.indexes)

    def __len__(self): # Number of batches per epoch
        return self.n // self.batch_size

    def index2rs(self, index): #From index of the sample to the number of the run and sample
        r = index // self.sampleperrun
        s = index % self.sampleperrun
        return r,s

    def __getitem__(self, index): # Generate one batch of data
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_r_s = [self.index2rs(k) for k in indexes]
        X, y = self.__data_generation(list_r_s)
        return (X, y)

    def on_epoch_end(self): # Updates indexes after each epoch
        self.indexes = np.arange(self.n)
        if self.shuffle == True:
            rng = np.random.default_rng()
            rng.shuffle(self.indexes)
    def __data_generation(self, list_r_s):  # Generates data containing batch_size samples

        X = np.full((self.batch_size, *self.dim, self.npredictors), np.nan)
        y = np.full((self.batch_size, *self.dim, self.ntargets), np.nan)

        for i, (r,s) in enumerate(list_r_s):
            fileIDX = os.path.join(self.path_data,f'{r+1+self.startrun:04d}',f'{s:03d}.nc')
            fileIDy = os.path.join(self.path_data,f'{r+1+self.startrun:04d}',f'{s+1:03d}.nc')
            ncx = netCDF4.Dataset(fileIDX, "r")
            ncy = netCDF4.Dataset(fileIDy, "r")
            for k in range(self.npredictors):
                varname, channel = featname2tuple(self.list_predictors[k])
                X[i,...,k] = ncx.variables[varname][0,channel,:,:]
            for k in range(self.ntargets):
                varname, channel = featname2tuple(self.list_targets[k])
                y[i,...,k] = ncy.variables[varname][0,channel,:,:]
            ncx.close()
            ncy.close()
        return (X, y)

