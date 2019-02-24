from matplotlib import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import cv2
import colorsys

class Filters():

    def cannyEdgeDetectorFilter(self, img):
        # Canny Libreria
        '''open_cv_image = cv2.fastNlMeansDenoising(open_cv_image, None, 3, 7, 21)
        _, open_cv_image = cv2.threshold(open_cv_image, 30, 255, cv2.THRESH_TOZERO)

        canny = cv2.Canny(open_cv_image, 100, 200)
        '''

        img = img.convert('L')
        open_cv_image = np.array(img)

        # 1) Applied Gaussian blur to reduce noise in the image
        cv2.GaussianBlur(open_cv_image, (15, 15), 0)

        # Step2: Calculating gradient magnitudes and directions

        kernel_size = 3
        sobelx = cv2.Sobel(open_cv_image, cv2.CV_64F, 1, 0, kernel_size)
        sobely = cv2.Sobel(open_cv_image, cv2.CV_64F, 0, 1, kernel_size)

        sobelx = np.uint8(np.absolute(sobelx))
        sobely = np.uint8(np.absolute(sobely))

        sobelCombined = cv2.bitwise_or(sobelx, sobely)
        #magnitudes = sqrt(sobelx** + sobely**2)


        #cv2.imshow("test", open_cv_image)
        img = Image.fromarray(open_cv_image)
        print("canny")
        return img

    def drogEdgeDetectorFilter(self, img):

        img = img.convert('L')
        open_cv_image = np.array(img)

        '''
        #cv2.GaussianBlur(open_cv_image, (15,15), 0)
        open_cv_image = cv2.fastNlMeansDenoising(open_cv_image, None, 3, 7, 21)
        _, img = cv2.threshold(open_cv_image, 30, 255, cv2.THRESH_TOZERO)
        sobelx = cv2.Sobel(open_cv_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        print('img drog:',open_cv_image[100,100])
        drog_img = sobelx + sobely

        cv2.imshow("test", drog_img)
        img = Image.fromarray(drog_img)
        
        '''

        '''
        px = img.load()
        print("######## px: ",px[100,100])
        width, height = img.size
        result = np.zeros((height, width, 2), dtype=np.uint8)

        # Create sobel x and y matrix
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        kernel_dim  = 1  # 3*3
        for i in range(kernel_dim, height - kernel_dim):
            for j in range(kernel_dim, width - kernel_dim):
                sum_x = 0
                sum_y = 0
                for n in range(0, 3):
                    for m in range(0, 3):
                        sum_x += px[m - 1 + j, n - 1 + i][0] * sobel_x[m, n]/4
                        sum_y += px[m - 1 + j, n - 1 + i][0] * sobel_y[m, n]/4


                result[i,j] = tuple([sum_x, 255])

        print(result[100, 100])
        '''

        width, height = img.size

        # Create sobel x and y matrix
        sobel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])

        kernel1 = np.zeros(open_cv_image.shape)
        kernel1[:sobel_x.shape[0], :sobel_x.shape[1]] = sobel_x
        kernel1 = np.fft.fft2(kernel1)

        kernel2 = np.zeros(open_cv_image.shape)
        kernel2[:sobel_y.shape[0], :sobel_y.shape[1]] = sobel_y
        kernel2 = np.fft.fft2(kernel2)


        im = np.array(open_cv_image)
        fim = np.fft.fft2(im)
        Gx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)
        Gy = np.real(np.fft.ifft2(kernel2 * fim)).astype(float)

        '''kernel_dim = 1  # 3*3
        result = np.array(img)

        for i in range(kernel_dim, height - kernel_dim):
            for j in range(kernel_dim, width - kernel_dim):
                sum_x = 0
                sum_y = 0
                for n in range(0, 3):
                    for m in range(0, 3):
                        sum_x += open_cv_image[n + j - 1, m + i - 1] * sobel_x[n, m] / 4
                        sum_y += open_cv_image[n + j - 1, m + i - 1] * sobel_y[n, m] / 4

                result[j, i] = abs(sum_x) + abs(sum_y)

        open_cv_image = np.array(result)
        '''
        open_cv_image = abs(Gx/4) + abs(Gy/4)
        img = Image.fromarray(open_cv_image)
        return img

    def harmonicMeanFilter(self, img):
        print("harmonic")
        return img

    def geometricMeanFilter(self, img):
        img = img.convert('RGB')

        '''
        px = img.load()
        width, height = img.size
        result = np.zeros((height, width, 3), dtype=np.uint8)

        kernel_dim = 4 #(*2+1)

        # kernel 9x9
        for i in range(kernel_dim, height - kernel_dim):
            for j in range(kernel_dim, width - kernel_dim):
                prod_red, prod_green, prod_blue = (1, 1, 1)
                for n in range(i - kernel_dim, i + kernel_dim):
                    for m in range(j - kernel_dim, j + kernel_dim):
                        r, g, b = px[m, n]
                        prod_red *= r
                        prod_green *= g
                        prod_blue *= b

                div = (kernel_dim * 2 + 1)**2
                result[i, j] = tuple([prod_red ** (1 / div), prod_green ** (1 / div), prod_blue ** (1 / div)])

        img = Image.fromarray(result, 'RGB')
        return img
        
        '''

        img = img.convert('RGB')
        open_cv_image = np.array(img)
        red = open_cv_image[:, :, 0]
        green = open_cv_image[:, :, 1]
        blue = open_cv_image[:, :, 2]

        width, height, _ = open_cv_image.shape

        mean_geometric = np.ones((9, 9))

        kernel1 = np.zeros((width, height))
        kernel1[:mean_geometric.shape[0], :mean_geometric.shape[1]] = mean_geometric
        kernel1 = np.fft.fft2(kernel1)

        im = np.array(red)
        fim = np.fft.fft2(im)
        Rx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)

        im = np.array(green)
        fim = np.fft.fft2(im)
        Gx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)

        im = np.array(blue)
        fim = np.fft.fft2(im)
        Bx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)

        open_cv_image[:, :, 0] = abs(Rx)
        open_cv_image[:, :, 1] = abs(Gx)
        open_cv_image[:, :, 2] = abs(Bx)

        img = Image.fromarray(open_cv_image)

        return img

    def arithmeticMeanFilter(self, img):

        ''' OLD IMPLEMENTATION
        px = img.load()
        width, height = img.size
        result = np.zeros((height, width, 3), dtype=np.uint8)

        kernel_dim = 1

        for i in range(kernel_dim, height - kernel_dim + 1):
            for j in range(kernel_dim, width - kernel_dim + 1):
                sum_red, sum_green, sum_blue = (0, 0, 0)
                for n in range(i - kernel_dim, i + kernel_dim):
                    for m in range(j - kernel_dim, j + kernel_dim):
                        r, g, b = px[m, n]
                        sum_red += r
                        sum_green += g
                        sum_blue += b

                result[i, j] = tuple([sum_red / 9, sum_green / 9, sum_blue / 9])

        '''

        img = img.convert('RGB')
        open_cv_image = np.array(img)
        red = open_cv_image[:, :, 0]
        green = open_cv_image[:, :, 1]
        blue = open_cv_image[:, :, 2]

        mean_arithmetic = np.ones((9, 9))*(1/81)

        width, height, _ = open_cv_image.shape

        kernel1 = np.zeros((width, height))
        kernel1[:mean_arithmetic.shape[0], :mean_arithmetic.shape[1]] = mean_arithmetic
        kernel1 = np.fft.fft2(kernel1)


        im = np.array(red)
        fim = np.fft.fft2(im)
        Rx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)

        im = np.array(green)
        fim = np.fft.fft2(im)
        Gx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)

        im = np.array(blue)
        fim = np.fft.fft2(im)
        Bx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)

        open_cv_image[:, :, 0] = abs(Rx)
        open_cv_image[:, :, 1] = abs(Gx)
        open_cv_image[:, :, 2] = abs(Bx)

        img = Image.fromarray(open_cv_image)

        return img

    def saturationFilter(self, index, img):

        img = img.convert('RGB')
        open_cv_image = np.array(img)

        # Same logic of luminance filter
        hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2HSV).astype("float32")
        s = hsv[:, :, 1]
        s += index
        s[s >= 255] = 255
        s[s <= 0] = 0

        hsv[:, :, 1] = s
        img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)
        img = Image.fromarray(img)
        return img


        '''
        img = img.convert('RGB')
        px = img.load()
        width, height = img.size

        result = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                r, g, b = px[j, i]

                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                s += index/255
                s = 0 if s <= 0 else s
                s = 1 if s >= 1 else s
                r, g, b = colorsys.hsv_to_rgb(h,s,v)
                rgb = self.mantainInRange(r,g,b)
                result[i, j] = rgb

        img = Image.fromarray(result, 'RGB')
        return img
        '''

    def luminanceFilter(self, index, img):

        img = img.convert('RGB')
        open_cv_image = np.array(img)

        # Change color space to use the Value of HSV to manipulate the luminance
        # the rest of code is similar to other color filter
        hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2HSV).astype("float32")
        v = hsv[:, :, 2]
        v += index
        v[v >= 255] = 255
        v[v <= 0] = 0

        hsv[:, :, 2] = v
        img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)
        img = Image.fromarray(img)
        return img


        '''
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
        
        '''



    def contrastFilter(self, index, img):

        img = img.convert('RGB')
        open_cv_image = np.array(img)

        # Select the three different channels
        red = open_cv_image[:, :, 0]
        green = open_cv_image[:, :, 1]
        blue = open_cv_image[:, :, 2]

        # factor used to apply the contrast function
        factor = (259.0 * (index + 255.0)) / (255.0 * (259.0 - index))

        # Calculate the new value for each pixel in the channel
        red = factor * (red - 128.0) + 128.0
        green = factor * (green - 128.0) + 128.0
        blue = factor * (blue - 128.0) + 128.0

        # Ensure that the value of pixels not go out of RGB range
        red[red >= 255] = 255
        red[red <= 0] = 0
        green[green >= 255] = 255
        green[green <= 0] = 0
        blue[blue >= 255] = 255
        blue[blue <= 0] = 0

        # Set the matrix image with the new arrays
        open_cv_image[:, :, 0] = red
        open_cv_image[:, :, 1] = green
        open_cv_image[:, :, 2] = blue

        # Convert the matrix to a PIL image that will show in the label
        img = Image.fromarray(open_cv_image)
        return img

        '''
        img = img.convert('RGB')

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
        '''

    def mantainInRange(self, red, green, blue):
        if red >= 255: red = 255
        if green >= 255: green = 255
        if blue >= 255: blue = 255
        if red <= 0: red = 0
        if green <= 0: green = 0
        if blue <= 0: blue = 0
        return tuple([red, green, blue])