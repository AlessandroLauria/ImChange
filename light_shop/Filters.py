from matplotlib import numpy as np
from PIL import Image
import colorsys
class Filters():

    def geometricMeanFilter(self,img):
        px = img.load()
        width, height = img.size
        result = np.zeros((height, width, 3), dtype=np.uint8)

        # kernel 9x9
        for i in range(4, height - 4):
            for j in range(4, width - 4):
                prod_red, prod_green, prod_blue = (1, 1, 1)
                for n in range(i - 4, i + 4):
                    for m in range(j - 4, j + 4):
                        r, g, b = px[m, n]
                        prod_red *= r
                        prod_green *= g
                        prod_blue *= b

                result[i, j] = tuple([prod_red ** (1 / 64), prod_green ** (1 / 64), prod_blue ** (1 / 64)])

        img = Image.fromarray(result, 'RGB')
        return img

    def arithmeticMeanFilter(self,img):
        px = img.load()
        width, height = img.size

        result = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(3, height - 3):
            for j in range(3, width - 3):
                sum_red, sum_green, sum_blue = (0, 0, 0)
                for n in range(i - 3, i + 3):
                    for m in range(j - 3, j + 3):
                        r, g, b = px[m, n]
                        sum_red += r
                        sum_green += g
                        sum_blue += b

                result[i, j] = tuple([sum_red / 49, sum_green / 49, sum_blue / 49])

        img = Image.fromarray(result, 'RGB')
        return img

    def saturationFilter(self, index, img):
        px = img.load()
        width, height = img.size

        result = np.zeros((height, width, 3), dtype=np.uint8)

        #r,g,b = px[100,100]
        #h, s, v = colorsys.rgb_to_hsv(r, g, b)
        #print("\nsaturation: ",h," ",s," ",v)

        for i in range(0, height):
            for j in range(0, width):
                r, g, b = px[j, i]

                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                s += index/255
                r, g, b = colorsys.hsv_to_rgb(h,s,v)
                rgb = self.mantainInRange(r,g,b)
                result[i, j] = rgb

        img = Image.fromarray(result, 'RGB')
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