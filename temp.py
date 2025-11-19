from PIL import Image
import numpy as np

image = Image.open("random.jpg")

array = np.empty((image.height, image.width))
blur_array = np.zeros((image.height, image.width))
kernel_blur = np.array([
  [1/9, 1/9, 1/9],
  [1/9, 1/9, 1/9], 
  [1/9, 1/9, 1/9]])

kernel_highpass = np.array([
  [-1/8, -1/8, -1/8],
  [-1/8, 1, -1/8], 
  [-1/8, -1/8, -1/8]])

kernel_laplacian = np.array([
  [0, -.1, 0],
  [-.1, 1+.4, -.1], 
  [-0, -.1, 0]])

kernel = kernel_laplacian

raster = image.load()

for y in range(image.height):
  for x in range(image.width):
    array[y,x] = (raster[x,y][0] + raster[x,y][1] + raster[x,y][2])/3

for y in range(image.height):
  for x in range(image.width):
    if x > 0 and x < image.width-1 and y > 0 and y < image.height-1:
      sum = 0
      for i in range(-1, 2):
        for j in range(-1, 2):
         sum += array[y+j, x+i] * kernel[j, i]
      blur_array[y,x] = sum





fourier_array = np.fft.fft2(blur_array)
shifted = np.fft.fftshift(fourier_array)

magnitude = np.log(np.abs(shifted)+1)
normalized = (magnitude/magnitude.max())*255

Image.fromarray(normalized.astype(np.uint8)).save("rebuild.jpg")
Image.fromarray(blur_array.astype(np.uint8)).save("blurred.jpg")
Image.fromarray(array.astype(np.uint8)).save("original.jpg")