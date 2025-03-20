import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pretrained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classifier
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(5, activation='softmax')(x)  # 5 face shape classes

model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generator for training
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'data/img_align_celeba/', target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')

val_generator = train_datagen.flow_from_directory(
    'data/img_align_celeba/', target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)
