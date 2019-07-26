import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
tf.compat.v1.enable_eager_execution()

import time

content_path = tf.keras.utils.get_file('turtle.jpg',
                                       'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
style_path = tf.keras.utils.get_file('kandinsky.jpg',
                                     'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


def load_image(img_path):
    max_dim = 512
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    print(long_dim)
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
    plt.show()


# for samoyed image painted in Udnie style by Francis Picabia:
content_image = load_image("./neural-style-transfer/content_samoyed.jpg")
style_image = load_image("./neural-style-transfer/style_Udnie_by_Francis_Picabia.jpg")

# for sea-turtle image results:
# content_image = load_image(content_path)
# style_image = load_image(style_path)

plt.subplot(1, 2, 1)
show_image(content_image, 'Content_Image')
plt.subplot(1, 2, 2)
show_image(style_image, 'Style_Image')
plt.show()

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
for layer in vgg.layers:
    print(layer.name)

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

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
image_result = tf.Variable(content_image)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.02, beta1=0.99, epsilon=1e-1)
style_weight = 1e-1
content_weight = 1e4


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


def clip_image(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def total_variation_loss(image):
    x_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]
    var_loss = tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)
    return var_loss


total_variation_weight = 1e6


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * total_variation_loss(image)

    gradient = tape.gradient(loss, image)
    optimizer.apply_gradients([(gradient, image)])
    image.assign(clip_image(image))


print("Commencing Style Transfer...\n")
startTime = time.time()
num_of_epochs = 10
steps_per_epoch = 10
for num in range(num_of_epochs):
    print("Epoch %d in progress..." % (num + 1))
    for s in range(steps_per_epoch):
        train_step(image_result)

    if num == num_of_epochs - 1:
        display.clear_output(wait=True)
        plt.imshow(image_result.read_value()[0])
        plt.show()

endTime = time.time()
print("Total Time: {:.1f}".format(endTime - startTime))
