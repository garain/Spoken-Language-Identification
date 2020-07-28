def create_cnn(height, width, depth,
    filters=(16, 32, 64), nb_classes):
    # initialize the input shape and channel dimension, assuming
     TensorFlow/channels-last ordering 
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
    	# if this is the first CONV layer then set the input appropriately
    	if i == 0:
    		x = inputs
    	# CONV => RELU => BN => POOL
    	x = Conv2D(f, (3, 3), 
    	        padding="same")(x)
    	x = Activation("relu")(x)
    	x = BatchNormalization(
    	            axis=chanDim)(x)
    	x = MaxPooling2D(
    	        pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(
            axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(nb_classes)(x)
    x = Activation("relu")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model
