import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, Input, Conv3DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from plots import PlotTimeSteppingPrediction


class ModelTimeSteppingPrediction:
    def __init__(self, input_shape, output_frames, learning_rate=1e-3):
        """
        Initialize the model configuration.
        
        Parameters:
        - input_shape (tuple): Shape of the input data (H, W, T, C)
        - output_frames (int): Number of frames to predict in the output
        - learning_rate (float): Learning rate for model compilation
        """
        self.input_shape = input_shape
        self.output_frames = output_frames
        self.learning_rate = learning_rate

    def create_conv_lstm(self):
        """
        Create a ConvLSTM model for predicting time-stepping frames.
        """
        model = Sequential()

        # ConvLSTM Layers
        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
                             input_shape=self.input_shape))
        model.add(BatchNormalization())

        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True))
        model.add(BatchNormalization())

        # Final Conv3D layer to generate the output frames
        model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same'))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae', metrics=['mae'])

        return model

    def create_unetlstm_model(self):
        """
        Create a UNet-LSTM model for time-stepping prediction.
        """
        inputs = Input(shape=self.input_shape)

        # Encoder path
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)

        # Bottleneck
        bottleneck = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
        bottleneck = BatchNormalization()(bottleneck)

        # Decoder path
        x = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3), padding='same', activation="relu")(bottleneck)
        x = concatenate([x, BatchNormalization()(bottleneck)])
        x = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), padding='same', activation="relu")(x)
        x = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), padding='same', activation="relu")(x)

        # Output layer
        outputs = Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)

        # Compile the model
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae', metrics=['mae'])

        return model

    def create_unet3d_model(self):
        inputs = Input(shape=self.input_shape)

        # **Encoder**
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        bn1 = BatchNormalization()(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization()(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(bn2)
        bn3 = BatchNormalization()(conv3)

        # **Bottleneck**
        bottleneck = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(bn3)
        bottleneck = BatchNormalization()(bottleneck)

        # **Decoder**
        up1 = Conv3DTranspose(128, (3, 3, 3), padding='same', activation="relu")(bottleneck)
        concat1 = concatenate([up1, bn3])

        up2 = Conv3DTranspose(64, (3, 3, 3), padding='same', activation="relu")(concat1)
        concat2 = concatenate([up2, bn2])

        up3 = Conv3DTranspose(32, (3, 3, 3), padding='same', activation="relu")(concat2)
        concat3 = concatenate([up3, bn1])

        # **Final Output**
        outputs = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(concat3)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

        return model
    

    def create_deep_unet3d_model(self):
        inputs = Input(shape=self.input_shape)

        # Encoder
        x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Bottleneck
        bottleneck = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
        bottleneck = BatchNormalization()(bottleneck)

        # Decoder
        x = Conv3DTranspose(128, (3, 3, 3), padding='same', activation="relu")(bottleneck)
        x = Conv3DTranspose(64, (3, 3, 3), padding='same', activation="relu")(x)
        x = Conv3DTranspose(32, (3, 3, 3), padding='same', activation="relu")(x)

        # Output layer
        outputs = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

        # Compile the model
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])

        return model


    def create_unet3d_multimodal(self):

        inputs = Input(shape=self.input_shape)

        # **Encoder**
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        bn1 = BatchNormalization()(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(bn1)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        bn2 = BatchNormalization()(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(bn2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        bn3 = BatchNormalization()(conv3)

        # **Bottleneck**
        bottleneck = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(bn3)
        bottleneck = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(bottleneck)
        bottleneck = BatchNormalization()(bottleneck)

        # **Decoder**
        up1 = Conv3DTranspose(128, (3, 3, 3), padding='same', activation="relu")(bottleneck)
        concat1 = concatenate([up1, bn3])
        up1 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(concat1)

        up2 = Conv3DTranspose(64, (3, 3, 3), padding='same', activation="relu")(up1)
        concat2 = concatenate([up2, bn2])
        up2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(concat2)

        up3 = Conv3DTranspose(32, (3, 3, 3), padding='same', activation="relu")(up2)
        concat3 = concatenate([up3, bn1])
        up3 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(concat3)

        # **Final Output - Only 1 Channel (vil)**
        outputs = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(up3)

        model = Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

        return model


    def train_model(self, model, X_train, y_train, X_test, y_test, epochs=50, batch_size=4):
        
        plotter = PlotTimeSteppingPrediction()

        # Visualize input and target frames
        plotter.visualize_input_target(X_train, y_train, sample_index=0)

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=1e-6)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stop, reduce_lr]
        )

        # Plot training loss
        plotter.plot_loss(history)

        return model, history
    