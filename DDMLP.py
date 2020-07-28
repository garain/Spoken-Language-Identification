 def Mod_voice(input_dim,nb_classes):
    model = Sequential()
    model.add(Dense(224, input_dim=input_dim))
    model.add(Activation("relu"))
    model.add(Dense(192))
    model.add(Activation("relu"))
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dense(96))
    model.add(Activation("relu"))
    model.add(Dense(160))
    model.add(Activation("relu"))
    model.add(Dropout(0.15))
    model.add(Dense(224))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(48))
    model.add(Activation("relu"))
    model.add(Dense(48))
    model.add(Activation("relu"))
    model.add(Dense(192))
    model.add(Activation("relu"))
    model.add(Dropout(0.15))
    model.add(Dense(192))
    model.add(Activation("relu"))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    
    model.compile(
    loss="categorical_crossentropy",
    optimizer="nadam",metrics=["acc"])
    
    return model
