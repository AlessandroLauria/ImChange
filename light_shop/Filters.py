from matplotlib import numpy as np
from PIL import Image

class Filters():

    def saturationFilter(self, index, img):
        px = img.load()
        width, height = img.size

        result = np.zeros((height, width, 3), dtype=np.uint8)
        div = 1

        for i in range(0, height):
            for j in range(0, width):
                hue, sat, value = px[j, i]
                sat += index / div
                if sat >= 255: sat = 255
                if sat <= 0: sat = 0
                rgb = tuple([hue, sat, value])
                result[i, j] = rgb

        img = Image.fromarray(result, 'HSV')
        img.convert("RGB")
        return img

    def luminanceFilter(self,index,img):
        px = img.load()
        width, height = img.size

        result = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                red, green, blue = px[j, i]
                red += index
                green += index
                blue += index
                rgb = self.mantainInRange(red, green, blue)
                result[i, j] = rgb

        img = Image.fromarray(result, 'RGB')
        return img

    def contrastFilter(self,index,img):

        px = img.load()
        width, height = img.size

        result = np.zeros((height, width, 3), dtype=np.uint8)

        factor = (259.0 * (index + 255.0)) / (255.0 * (259.0 - index))

        for i in range(0, height):
            for j in range(0, width):
                red, green, blue = px[j,i]
                red = factor * (red - 128.0) + 128.0
                green = factor * (green - 128.0) + 128.0
                blue = factor * (blue - 128.0) + 128.0
                rgb = self.mantainInRange(red, green, blue)
                result[i, j] = rgb

        img = Image.fromarray(result, 'RGB')
        return img

    # Evito che i valori RGB sforino il range
    def mantainInRange(self, red, green, blue):
        if red >= 255: red = 255
        if green >= 255: green = 255
        if blue >= 255: blue = 255
        if red <= 0: red = 0
        if green <= 0: green = 0
        if blue <= 0: blue = 0
        return tuple([red, green, blue])