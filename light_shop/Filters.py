from matplotlib import numpy as np
from PIL import Image, ImageFilter
from scipy.fftpack import fftfreq
import cv2

class Filters():

    def translate(self, img, x, y):
        img = img.convert('RGB')
        open_cv_image = np.array(img)

        red = open_cv_image[:, :, 0]
        green = open_cv_image[:, :, 1]
        blue = open_cv_image[:, :, 2]

        shift_rows, shift_cols = (x, y)
        nr, nc = img.size[1], img.size[0]
        Nr, Nc = fftfreq(nr), fftfreq(nc)
        Nc, Nr = np.meshgrid(Nc, Nr)

        fft_inputarray = np.fft.fft2(red)
        fourier_shift = np.exp(-1j * 2 * np.pi * ((shift_rows * Nr) + (shift_cols * Nc))/200)
        red = np.fft.ifft2(fft_inputarray * fourier_shift)

        fft_inputarray = np.fft.fft2(green)
        fourier_shift = np.exp(-1j * 2 * np.pi * ((shift_rows * Nr) + (shift_cols * Nc))/200)
        green = np.fft.ifft2(fft_inputarray * fourier_shift)

        fft_inputarray = np.fft.fft2(blue)
        fourier_shift = np.exp(-1j * 2 * np.pi * ((shift_rows * Nr) + (shift_cols * Nc))/200)
        blue = np.fft.ifft2(fft_inputarray * fourier_shift)

        open_cv_image[:, :, 0] = red
        open_cv_image[:, :, 1] = green
        open_cv_image[:, :, 2] = blue

        open_cv_image = abs(open_cv_image)
        img = Image.fromarray(open_cv_image)
        return img

    def scaling(self, img, index):

        print("scaling")

    def statisticalRegionMerging(self, img):
        print("srm")

    def cannyEdgeDetectorFilter(self, img):

        img = img.convert('L')
        im = np.array(img, dtype=float)  # Convert to float to prevent clipping values

        # 1) Applied Gaussian blur to reduce noise in the image
        cv2.GaussianBlur(im, (15, 15), 0)

        # 2) Calculate gradient magnitudes and directions
        kernel_size = 3
        sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, kernel_size)
        sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, kernel_size)

        magnitude = np.sqrt(sobelx**2 + sobely**2)

        angles = np.arctan2(sobely, sobelx)

        # 3) Non maximum suppression
        mag_sup = magnitude.copy()

        for x in range(1, im.shape[0] - 1):
            for y in range(1, im.shape[1] - 1):
                if angles[x][y] == 0:
                    if (magnitude[x][y] <= magnitude[x][y + 1]) or \
                            (magnitude[x][y] <= magnitude[x][y - 1]):
                        mag_sup[x][y] = 0
                elif angles[x][y] == 45:
                    if (magnitude[x][y] <= magnitude[x - 1][y + 1]) or \
                            (magnitude[x][y] <= magnitude[x + 1][y - 1]):
                        mag_sup[x][y] = 0
                elif angles[x][y] == 90:
                    if (magnitude[x][y] <= magnitude[x + 1][y]) or \
                            (magnitude[x][y] <= magnitude[x - 1][y]):
                        mag_sup[x][y] = 0
                else:
                    if (magnitude[x][y] <= magnitude[x + 1][y + 1]) or \
                            (magnitude[x][y] <= magnitude[x - 1][y - 1]):
                        mag_sup[x][y] = 0

        # 4) Thresholding

        m = np.max(mag_sup)
        th = 0.2 * m
        tl = 0.1 * m

        width, height = im.shape

        gnh = np.zeros((width, height))
        gnl = np.zeros((width, height))

        for x in range(width):
            for y in range(height):
                if mag_sup[x][y] >= th:
                    gnh[x][y] = mag_sup[x][y]
                if mag_sup[x][y] >= tl:
                    gnl[x][y] = mag_sup[x][y]
        #gnh = gnl + gnh

        '''

        def traverse(i, j):
            x = [-1, 0, 1, -1, 1, -1, 0, 1]
            y = [-1, -1, -1, 0, 0, 1, 1, 1]
            for k in range(8):
                if gnh[i + x[k]][j + y[k]] == 0 and gnl[i + x[k]][j + y[k]] != 0:
                    gnh[i + x[k]][j + y[k]] = 1
                    traverse(i + x[k], j + y[k])

        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if gnh[i][j]:
                    gnh[i][j] = 1
                    traverse(i, j)

        '''

        img = Image.fromarray(gnh)
        img = img.convert('L')
        return img


    def drogEdgeDetectorFilter(self, img):

        img = img.convert('L')
        open_cv_image = np.array(img)

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

        open_cv_image = abs(Gx/4) + abs(Gy/4)
        img = Image.fromarray(open_cv_image)
        return img

    def harmonicMeanFilter(self, img):
        print("harmonic")
        return img

    def geometricMeanFilter(self, img):

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

        img = img.convert('L')

        width, height = img.size
        open_cv_image = np.array(img)

        w = 1  # (*2+1)

        for i in range(w, height - w):
            for j in range(w, width - w):
                block_red = open_cv_image[i - w:i + w + 1, j - w:j + w + 1]
                m_r = np.prod(block_red, dtype=np.float32)**(1/len(block_red))
                open_cv_image[i][j] = int(m_r)


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
        
        '''

        img = img.convert('RGB')
        im = np.array(img)
        red = im[:, :, 0]
        green = im[:, :, 1]
        blue = im[:, :, 2]
        w = 2

        for i in range(w, im.shape[0] - w):
            for j in range(w, im.shape[1] - w):
                block_red = red[i - w:i + w + 1, j - w:j + w + 1]
                m_r = np.mean(block_red, dtype=np.float32)
                red[i][j] = int(m_r)

                block_green = green[i - w:i + w + 1, j - w:j + w + 1]
                m_g = np.mean(block_green, dtype=np.float32)
                green[i][j] = int(m_g)

                block_blue = blue[i - w:i + w + 1, j - w:j + w + 1]
                m_r = np.mean(block_blue, dtype=np.float32)
                blue[i][j] = int(m_r)

        im[:, :, 0] = red
        im[:, :, 1] = green
        im[:, :, 2] = blue
        img = Image.fromarray(im)
        img = img.convert('RGB')
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