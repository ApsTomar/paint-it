import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
tf.compat.v1.enable_eager_execution()

import numpy as np
import time
import functools
import time


# content_path = tf.keras.utils.get_file('turtle.jpg',
#                                        'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
# style_path = tf.keras.utils.get_file('kandinsky.jpg',
#                                      'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


def load_image(img_path):
    max_dim = 512
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def show_image(img, title=None):
    if len(img.shape) > 3:
        img = tf.squeeze(img, axis=0)
    plt.imshow(img)
    if title:
        plt.title = title


content_image = load_image("./GOT_Doggo.png")
style_image = load_image("./The Swing by Jean-Honoré Fragonard.png")
plt.subplot(1, 2, 1)
show_image(content_image, 'Content_Image')
plt.subplot(1, 2, 2)
show_image(style_image, 'Style_Image')
plt.show()

# preprocess_input = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
# preprocess_input = tf.image.resize(input, (224, 224))

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
# for layer in vgg.layers:
#     print(layer.name)

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def vgg_layers(layers_name):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layers_name]
    model = tf.keras.Model([vgg.input], outputs)
    return model


# style_extractor = vgg_layers(style_layers)
# style_outputs = style_extractor(style_image * 255)


def gram_matrix(style_tensor):
    output = tf.linalg.einsum('bijc,bijd->bcd', style_tensor, style_tensor)
    dimension = tf.shape(output)
    num_of_locations = tf.cast(dimension[1] * dimension[2], tf.float32)
    return output / num_of_locations


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs, training=None, mask=None):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}


extractor = StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content_image))
# print('Styles:')
# for name, output in sorted(results['style'].items()):
#     print("  ", name)
#     print("    shape: ", output.numpy().shape)
#     print("    min: ", output.numpy().min())
#     print("    max: ", output.numpy().max())
#     print("    mean: ", output.numpy().mean())
#     print()
#
# print("Contents:")
# for name, output in sorted(results['content'].items()):
#     print("  ", name)
#     print("    shape: ", output.numpy().shape)
#     print("    min: ", output.numpy().min())
#     print("    max: ", output.numpy().max())
#     print("    mean: ", output.numpy().mean())

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
image_result = tf.Variable(content_image)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.02, beta1=0.99, epsilon=1e-1)
style_weight = 1e-2
content_weight = 1e4


def clip_image(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(extracted_output):
    style_outputs = extracted_output['style']
    content_outputs = extracted_output['content']
    style_loss = tf.add_n(
        [tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
    content_loss = tf.add_n(
        [tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    content_loss *= content_weight / num_content_layers
    total_loss = style_loss + content_loss
    return total_loss


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    gradient = tape.gradient(loss, image)
    optimizer.apply_gradients([(gradient, image)])
    image.assign(clip_image(image))


# train_step(image_result)
# train_step(image_result)
# train_step(image_result)
# plt.imshow(image_result.read_value()[0])
# plt.show()

start = time.time()
num_of_epochs = 10
steps_per_epoch = 10
step_count = 0
for num in range(num_of_epochs):
    for s in range(steps_per_epoch):
        train_step(image_result)
        step_count += 1
        print("Training epoch %d in progress" % num)

    display.clear_output(wait=True)
    plt.imshow(image_result.read_value())
    plt.title("Training epoch: {}".format(num))
    plt.show()

end = time.time()
print("Total Time: {:.1f}".format(end - start))