from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

model = load_model('autoencoders.h5')
print(model.summary())
test_image = image.load_img('testImage.png')
plt.imshow(test_image)
plt.show()
test_image = test_image.resize((28,28))
test_image = image.img_to_array(test_image)
test_image = test_image.astype('float32') / 255.
test_image = np.delete(test_image, [0] ,axis=2)
test_image = np.delete(test_image, [0] ,axis=2)
test_image = np.reshape(test_image, (1, 28, 28, 1))
print(test_image.shape)
test_image = model.predict(test_image)
test_image = np.reshape(test_image, (28, 28))
print(test_image.shape)
plt.imshow(test_image)
plt.show()