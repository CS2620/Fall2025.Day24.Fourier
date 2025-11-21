# Code from class
from PIL import Image
import numpy as np

image = Image.open("ducks.jpg")
raster = image.load()

array = np.empty((image.height, image.width))

kernel = np.array([[0, 0, 0],
                   [-1/2, 1, 1/2],
                   [0, 0, 0]])


for y in range(image.height):
    for x in range(image.width):
        r, g, b, *_ = raster[x,y]
        k = (r + g + b)//3
        raster[x,y] = (k, k, k)
        
for y in range(image.height):
    for x in range(image.width):
        
        if x < 1 or x >= image.width-1 or y < 1 or y >= image.height-1:
            array[y,x] = raster[x,y][0]
        else:
            sum = 0
            for i in range(3):
                for j in range(3):
                    k = raster[x+i-1, y+j-1][0] * kernel[i,j]   
                    sum += k     
            
            
            array[y,x] = int(sum)
        
fourier = np.fft.fft2(array)

for y in range(image.height):
    for x in range(image.width):
        f = fourier[y,x]
        
        x_diff = abs(x - image.width/2)
        y_diff = abs(y - image.height/2)
        
        amount = 214
        
        if not(x_diff < amount or y_diff < amount):
            fourier[y,x] = 0


shifted = np.fft.fftshift(fourier)

magnitude = np.log(np.abs(shifted)+1)
normalized = magnitude/magnitude.max()*255

inverted = np.fft.ifft2(fourier)
abs_inverted = np.abs(inverted)

# print(fourier[0])
        
Image.fromarray(array.astype(np.uint8)).save("final.png")
Image.fromarray(normalized.astype(np.uint8)).save("fourier.png")
Image.fromarray(abs_inverted.astype(np.uint8)).save("inverted.png")
