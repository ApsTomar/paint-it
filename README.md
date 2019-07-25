# paint-it

   How about we paint our images in similar manner as Van Gogh painted Starry Night! Thanks to Neural Style Transfer, it is quite possible.
   The model uses pretrained VGG-19 network for image features extraction. The content image is our original image which is to be painted same as the style image. We use lower conv layers of VGG-19 for style extraction as it gives more details of how the style texture is constructed. For content, higher conv layers are used as they retain the spatial information of the image which is vital for content image. Mean Square Error is computed by comparing Gram Matrix of the resulting image and style image, which is then optimized in order to gnerate the image in same style.
