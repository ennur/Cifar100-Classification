
import os,shutil

original_dataset_train_dir = 'C:\\Users\ennur\Desktop\cifar-100\Train'
original_dataset_test_dir='C:\\Users\ennur\Desktop\cifar-100\Test'

kategoriler=['beetle','bus','house','lamp','sunflower','turtle'] # 7,13,37,40,82,93

base_dir="C:\\Users\harun\Desktop\classification_cifar100"
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'Train')
os.mkdir(train_dir)
test_dir = os.path.join(base_dir, 'Test')
os.mkdir(test_dir)

train_beetle_dir = os.path.join(train_dir, 'beetle')
os.mkdir(train_beetle_dir)
train_bus_dir = os.path.join(train_dir, 'bus')
os.mkdir(train_bus_dir)
train_house_dir = os.path.join(train_dir, 'house')
os.mkdir(train_house_dir)
train_lamp_dir = os.path.join(train_dir, 'lamp')
os.mkdir(train_lamp_dir)
train_sunflower_dir = os.path.join(train_dir, 'sunflower')
os.mkdir(train_sunflower_dir)
train_turtle_dir = os.path.join(train_dir, 'turtle')
os.mkdir(train_turtle_dir)

test_beetle_dir = os.path.join(test_dir, 'beetle')
os.mkdir(test_beetle_dir)
test_bus_dir = os.path.join(test_dir, 'bus')
os.mkdir(test_bus_dir)
test_house_dir = os.path.join(test_dir, 'house')
os.mkdir(test_house_dir)
test_lamp_dir = os.path.join(test_dir, 'lamp')
os.mkdir(test_lamp_dir)
test_sunflower_dir = os.path.join(test_dir, 'sunflower')
os.mkdir(test_sunflower_dir)
test_turtle_dir = os.path.join(test_dir, 'turtle')
os.mkdir(test_turtle_dir)

fnames_beetle = ['beetle_s_{0:06d}.png'.format(i)for i in range(1,3000)]
fnames_ladybug = ['ladybug_s_{0:06d}.png'.format(i)for i in range(1,2000)]
fnames_tiger_beetle=['tiger_beetle_s_{0:06d}.png'.format(i)for i in range(1,2000)]
fnames_ladybeetle=['ladybeetle_s_{0:06d}.png'.format(i)for i in range(1,100)]
fnames= fnames_beetle+ fnames_ladybug+fnames_tiger_beetle+fnames_ladybeetle

for fname in fnames:
    try:
        src_train = os.path.join(original_dataset_train_dir, fname)
        dst_train = os.path.join(train_beetle_dir, fname)
        shutil.copyfile(src_train, dst_train)
    except(FileNotFoundError):
        continue
for fname in fnames:
    try:
        src_test = os.path.join(original_dataset_test_dir, fname)
        dst_test = os.path.join(test_beetle_dir, fname)
        shutil.copyfile(src_test, dst_test)
    except(FileNotFoundError):
        continue

fnames_bus = ['bus_s_{0:06d}.png'.format(i)for i in range(1,4000)]
fnames_minibus = ['minibus_s_{0:06d}.png'.format(i)for i in range(1,2500)]
fnames_school_bus=['school_bus_s_{0:06d}.png'.format(i)for i in range(1,2500)]
fnames_trolleybus=['trolleybus_s_{0:06d}.png'.format(i)for i in range(1,500)]

fnames= fnames_bus+fnames_minibus+fnames_school_bus+fnames_trolleybus

for fname in fnames:
    try:
        src_train = os.path.join(original_dataset_train_dir, fname)
        dst_train = os.path.join(train_bus_dir, fname)
        shutil.copyfile(src_train, dst_train)
    except(FileNotFoundError):
        continue
for fname in fnames:
    try:
        src_test = os.path.join(original_dataset_test_dir, fname)
        dst_test = os.path.join(test_bus_dir, fname)
        shutil.copyfile(src_test, dst_test)
    except(FileNotFoundError):
        continue

fnames_house = ['house_s_{0:06d}.png'.format(i)for i in range(1,1500)]
fnames_boarding_house = ['boarding_house_s_{0:06d}.png'.format(i)for i in range(1,2500)]
fnames_bungalow=['bungalow_s_{0:06d}.png'.format(i)for i in range(1,600)]

fnames= fnames_house+fnames_boarding_house+fnames_bungalow

for fname in fnames:
    try:
        src_train = os.path.join(original_dataset_train_dir, fname)
        dst_train = os.path.join(train_house_dir, fname)
        shutil.copyfile(src_train, dst_train)
    except(FileNotFoundError):
        continue
for fname in fnames:
    try:
        src_test = os.path.join(original_dataset_test_dir, fname)
        dst_test = os.path.join(test_house_dir, fname)
        shutil.copyfile(src_test, dst_test)
    except(FileNotFoundError):
        continue

fnames_candle = ['candle_s_{0:06d}.png'.format(i)for i in range(1,400)]
fnames_discharge_lamp = ['discharge_lamp_s_{0:06d}.png'.format(i)for i in range(1,2000)]
fnames_electric_lamp=['electric_lamp_s_{0:06d}.png'.format(i)for i in range(1,800)]
fnames_lamp=['lamp_s_{0:06d}.png'.format(i)for i in range(1,2500)]
fnames_rush_candle=['rush_candle_s_{0:06d}.png'.format(i)for i in range(1,300)]
fnames_rushlight=['rushlight_s_{0:06d}.png'.format(i)for i in range(1,600)]
fnames_wax_light=['wax_light_s_{0:06d}.png'.format(i)for i in range(1,16)]

fnames= fnames_candle+ fnames_discharge_lamp+ fnames_electric_lamp+ fnames_lamp+ fnames_rush_candle+fnames_rushlight+fnames_wax_light

for fname in fnames:
    try:
        src_train = os.path.join(original_dataset_train_dir, fname)
        dst_train = os.path.join(train_lamp_dir, fname)
        shutil.copyfile(src_train, dst_train)
    except(FileNotFoundError):
        continue
for fname in fnames:
    try:
        src_test = os.path.join(original_dataset_test_dir, fname)
        dst_test = os.path.join(test_lamp_dir, fname)
        shutil.copyfile(src_test, dst_test)
    except(FileNotFoundError):
        continue

fnames_sunflower = ['sunflower_s_{0:06d}.png'.format(i)for i in range(1,2500)]

for fname in fnames_sunflower:
    try:
        src_train = os.path.join(original_dataset_train_dir, fname)
        dst_train = os.path.join(train_sunflower_dir, fname)
        shutil.copyfile(src_train, dst_train)
    except(FileNotFoundError):
        continue
for fname in fnames_sunflower:
    try:
        src_test = os.path.join(original_dataset_test_dir, fname)
        dst_test = os.path.join(test_sunflower_dir, fname)
        shutil.copyfile(src_test, dst_test)
    except(FileNotFoundError):
        continue

fnames_marine_turtle = ['marine_turtle_s_{0:06d}.png'.format(i)for i in range(1,2000)]
fnames_sea_turtle = ['sea_turtle_s_{0:06d}.png'.format(i)for i in range(1,2800)]
fnames_turtle=['turtle_s_{0:06d}.png'.format(i)for i in range(1,2500)]

fnames= fnames_turtle+ fnames_sea_turtle+fnames_marine_turtle

for fname in fnames:
    try:
        src_train = os.path.join(original_dataset_train_dir, fname)
        dst_train = os.path.join(train_turtle_dir, fname)
        shutil.copyfile(src_train, dst_train)
    except(FileNotFoundError):
        continue
for fname in fnames:
    try:
        src_test = os.path.join(original_dataset_test_dir, fname)
        dst_test = os.path.join(test_turtle_dir, fname)
        shutil.copyfile(src_test, dst_test)
    except(FileNotFoundError):
        continue

train_dir="C:\\Users\ennur\Desktop\classification_cifar100\Train"
test_dir="C:\\Users\ennur\Desktop\classification_cifar100\Test"

#train_datagen = ImageDataGenerator(rescale=1./255)

from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(32,32),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(32,32),
        batch_size=20,
        class_mode='categorical')


from keras import layers,models,optimizers

model=models.Sequential()
model.add(layers.Conv2D(16, (2, 2), activation='relu', padding='same',
                        input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (2, 2),padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (2, 2),padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (2, 2),padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.01),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=150,
      epochs=80,
      validation_data=validation_generator,
      validation_steps=30,
      verbose=2
      )

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
