from matplotlib import numpy as np
from PIL import Image, ImageFilter
from scipy.fftpack import fftfreq
from SRM import SRM
import imutils
import cv2
from scipy import signal

class Filters():

    def rotate_coords(self, x, y, theta, ox, oy):

        s, c = np.sin(theta), np.cos(theta)
        x, y = np.asarray(x) - ox, np.asarray(y) - oy
        return x * c - y * s + ox, x * s + y * c + oy

    def rotate(self, img, angle, fill=0, resize=True):

        # Images have origin at the top left, so negate the angle.
        theta = angle * np.pi / 180

        # Dimensions of source image. Note that scipy.misc.imread loads
        # images in row-major order, so src.shape gives (height, width).
        img = img.convert('RGB')
        src = np.array(img)
        red = src[:, :, 0]
        green = src[:, :, 1]
        blue = src[:, :, 2]

        sh, sw = src.shape[0], src.shape[1]

        ox, oy = sh/2, sw/2

        # Rotated positions of the corners of the source image.
        cx, cy = self.rotate_coords([0, sw, sw, 0], [0, 0, sh, sh], theta, ox, oy)

        # Determine dimensions of destination image.
        dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))

        # Coordinates of pixels in destination image.
        dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))

        # Corresponding coordinates in source image. Since we are
        # transforming dest-to-src here, the rotation is negated.
        sx, sy = self.rotate_coords(dx + cx.min(), dy + cy.min(), -theta, ox, oy)

        # Select nearest neighbour.
        sx, sy = sx.round().astype(int), sy.round().astype(int)

        # Mask for valid coordinates.
        mask = (0 <= sx) & (sx < sw) & (0 <= sy) & (sy < sh)

        # Create destination image.
        red_dest = np.empty(shape=(dh, dw), dtype=src.dtype)
        green_dest = np.empty(shape=(dh, dw), dtype=src.dtype)
        blue_dest = np.empty(shape=(dh, dw), dtype=src.dtype)

        # Copy valid coordinates from source image.
        red_dest[dy[mask], dx[mask]] = red[sy[mask], sx[mask]]
        green_dest[dy[mask], dx[mask]] = green[sy[mask], sx[mask]]
        blue_dest[dy[mask], dx[mask]] = blue[sy[mask], sx[mask]]

        # Fill invalid coordinates.
        red_dest[dy[~mask], dx[~mask]] = fill
        green_dest[dy[~mask], dx[~mask]] = fill
        blue_dest[dy[~mask], dx[~mask]] = fill

        src = np.resize(src, (dh, dw, 3))
        src[:, :, 0] = red_dest
        src[:, :, 1] = green_dest
        src[:, :, 2] = blue_dest

        if not resize:
            new_src = np.array(img)
            new_src = imutils.rotate(new_src, angle)
            img = Image.fromarray(new_src.astype('uint8'))

        else:
            img = Image.fromarray(src.astype('uint8'))

        return img

    '''
    def rotate(self, img, angle=0):

        img = img.convert('L')
        im = np.array(img)

        width, height = img.size

        im_rot = np.zeros((height*2,width*2))

        angle = -10

        for i in range(width):
            for j in range(height):
                im_rot[int(j*np.cos(angle) + i*np.sin(angle)) + int(width/2), int(-j*np.sin(angle) + i*np.cos(angle)) + int(height/2)] = im[j, i]
               # print("im_rot: [",int(j*np.cos(angle) + i*np.sin(angle)),",",int(-j*np.sin(angle) + i*np.cos(angle))," im: [",i,",",j,"]")

        img = Image.fromarray(im_rot)
        return img
    '''

    def translate(self, img, x, y):
        img = img.convert('RGB')
        open_cv_image = np.array(img)

        red = open_cv_image[:, :, 0]
        green = open_cv_image[:, :, 1]
        blue = open_cv_image[:, :, 2]

        '''
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
        '''

        x = -x
        red = np.roll(red, x, axis=0)
        if x >= 0: red[:x, :] = 0
        else: red[img.height+x:, :] = 0

        red = np.roll(red, y, axis=1)
        if y >= 0: red[:, :y] = 0
        else: red[:, img.width+y:] = 0

        green = np.roll(green, x, axis=0)
        if x >= 0: green[:x, :] = 0
        else: green[img.height+x:, :] = 0

        green = np.roll(green, y, axis=1)
        if y >= 0: green[:, :y] = 0
        else: green[:, img.width+y:] = 0

        blue = np.roll(blue, x, axis=0)
        if x >= 0: blue[:x, :] = 0
        else: blue[img.height+x:, :] = 0

        blue = np.roll(blue, y, axis=1)
        if y >= 0: blue[:, :y] = 0
        else: blue[:, img.width+y:] = 0

        open_cv_image[:, :, 0] = red
        open_cv_image[:, :, 1] = green
        open_cv_image[:, :, 2] = blue

        open_cv_image = abs(open_cv_image)
        img = Image.fromarray(open_cv_image)
        return img

    def scaling(self, img, index, resize=False):

        img = img.convert('RGB')
        im = np.array(img)
        red = im[:, :, 0]
        green = im[:, :, 1]
        blue = im[:, :, 2]

        width, height = img.size

        # Calculate the second index value to maintain the aspect ratio
        aspect_ratio = width/height
        index1 = index

        index2 = int((index1 + width - (aspect_ratio*height))/aspect_ratio)

        new_img = img.resize((width - index1, height - index2))
        new_img = np.array(new_img)

        new_red = np.zeros((height - index2, width - index1))
        new_green = np.zeros((height - index2, width - index1))
        new_blue = np.zeros((height - index2, width - index1))

        if index > 0 :
            new_red[:, :] = red[0: height - index2, 0:width - index1]
            new_green[:, :] = green[0: height - index2, 0:width - index1]
            new_blue[:, :] = blue[0: height - index2, 0:width - index1]
        else:
            new_red[0:height, 0:width] = red[:, :]
            new_green[0:height, 0:width] = green[:, :]
            new_blue[0:height, 0:width] = blue[:, :]

        new_img[:, :, 0] = new_red
        new_img[:, :, 1] = new_green
        new_img[:, :, 2] = new_blue
        if not resize:
            img = Image.fromarray(new_img)
            img = img.resize((width, height))

        else:
            img = img.resize((width + index1, height + index2))

        return img

    def statisticalRegionMerging(self, img, n_sample):

        img = img.convert('RGB')
        image = np.array(img)

        '''
        width, height = img.size

        final_image = []

        for i in range(1, width - 1):
            for j in range(1, height - 1):
                #mean = (image[j, i] + image[j + 1, i] +image[j - 1, i] + image[j, i + 1] + image[j, i - 1]) / 5
                mean = image[j, i]
                for m in range(i + 1, width - 1):
                    for n in range(j + 1, height - 1):

                        if ((mean + image[n, m])/2 > mean - 50) and ((mean + image[n, m])/2 < mean + 50):
                            mean = (mean + image[n, m])/2
                            image[j, i] = mean
                            image[n, m] = mean
        
        '''

        srm = SRM(image, n_sample)
        segmented = srm.run()
        segmented = np.array(segmented).astype(int)

        red = segmented[:, :, 0]
        green = segmented[:, :, 1]
        blue = segmented[:, :, 2]

        image[:, :, 0] = red
        image[:, :, 1] = green
        image[:, :, 2] = blue

        #cv2.imshow("test", segmented)

        #print (segmented)

        img = Image.fromarray(image)
        return img

    def cannyEdgeDetectorFilter(self, img, sigma=1, t_low=0.01, t_high=0.2):

        im = img.convert('L')
        img = np.array(im, dtype=float)
        '''
        # 1) Applied Gaussian blur to reduce noise in the image
        cv2.GaussianBlur(im, (3, 3), 0)

        # 2) Calculate gradient magnitudes and directions
        kernel_size = 11
        sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, kernel_size)
        sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, kernel_size)

        magnitude = np.sqrt(sobelx**2 + sobely**2)

        magnitude = (magnitude.astype('float') - magnitude.min()) / (magnitude.max() - magnitude.min())

        angles = np.arctan2(sobely, sobelx)

        #cv2.imshow("mag", magnitude)

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

        #cv2.imshow("sup", mag_sup)

        # 4) Thresholding

        m = np.max(mag_sup)
        #th = 0.15 * m
        #tl = 0.09 * m
        th = 0.15
        tl = 0.09

        width, height = im.shape

        gnh = np.zeros((width, height))
        gnl = np.zeros((width, height))

        for x in range(width):
            for y in range(height):
                if mag_sup[x][y] >= th:
                    gnh[x][y] = mag_sup[x][y]
                if mag_sup[x][y] >= tl:
                    gnl[x][y] = mag_sup[x][y]
        gnh = gnl + gnh

        # edge linking

        for x in range(width - 2):
            for y in range(height - 2):
                if gnl[x, y] >= 1 and ((gnh[x+1, y] >= 1) or (gnh[x, y+1] >= 1) or (gnh[x-1, y] >= 1) or (gnh[x, y-1] >= 1)\
                                       or (gnh[x+1, y+1] >= 1) or (gnh[x+1, y-1] >= 1) or (gnh[x-1, y-1] >= 1) or (gnh[x-1, y+1] >= 1)):
                    gnh[x, y] = gnl[x, y]
                #else: gnh[x, y] = 0

        '''
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

        # 1) Convolve gaussian kernel with gradient
        # gaussian kernel
        halfSize = 3 * sigma
        maskSize = 2 * halfSize + 1
        mat = np.ones((maskSize, maskSize)) / (float)(2 * np.pi * (sigma ** 2))
        xyRange = np.arange(-halfSize, halfSize + 1)
        xx, yy = np.meshgrid(xyRange, xyRange)
        x2y2 = (xx ** 2 + yy ** 2)
        exp_part = np.exp(-(x2y2 / (2.0 * (sigma ** 2))))
        gSig = mat * exp_part

        gx, gy = self.drogEdgeDetectorFilter(gSig, ret_grad=True, pillow=False)

        # 2) Magnitude and Angles
        # apply kernels for Ix & Iy
        Ix = cv2.filter2D(img, -1, gx)
        Iy = cv2.filter2D(img, -1, gy)

        # compute magnitude
        mag = np.sqrt(Ix ** 2 + Iy ** 2)

        # normalize magnitude image
        normMag = my_Normalize(mag)

        # compute orientation of gradient
        orient = np.arctan2(Iy, Ix)

        # round elements of orient
        orientRows = orient.shape[0]
        orientCols = orient.shape[1]

        # 3) Non maximum suppression
        for i in range(0, orientRows):
            for j in range(0, orientCols):
                if normMag[i, j] > t_low:
                    # case 0
                    if (orient[i, j] > (- np.pi / 8) and orient[i, j] <= (np.pi / 8)):
                        orient[i, j] = 0
                    elif (orient[i, j] > (7 * np.pi / 8) and orient[i, j] <= np.pi):
                        orient[i, j] = 0
                    elif (orient[i, j] >= -np.pi and orient[i, j] < (-7 * np.pi / 8)):
                        orient[i, j] = 0
                    # case 1
                    elif (orient[i, j] > (np.pi / 8) and orient[i, j] <= (3 * np.pi / 8)):
                        orient[i, j] = 3
                    elif (orient[i, j] >= (-7 * np.pi / 8) and orient[i, j] < (-5 * np.pi / 8)):
                        orient[i, j] = 3
                    # case 2
                    elif (orient[i, j] > (3 * np.pi / 8) and orient[i, j] <= (5 * np.pi / 8)):
                        orient[i, j] = 2
                    elif (orient[i, j] >= (-5 * np.pi / 4) and orient[i, j] < (-3 * np.pi / 8)):
                        orient[i, j] = 2
                    # case 3
                    elif (orient[i, j] > (5 * np.pi / 8) and orient[i, j] <= (7 * np.pi / 8)):
                        orient[i, j] = 1
                    elif (orient[i, j] >= (-3 * np.pi / 8) and orient[i, j] < (-np.pi / 8)):
                        orient[i, j] = 1


        mag = normMag
        mag_thin = np.zeros(mag.shape)
        for i in range(mag.shape[0] - 1):
            for j in range(mag.shape[1] - 1):
                if mag[i][j] < t_low:
                    continue
                if orient[i][j] == 0:
                    if mag[i][j] > mag[i][j - 1] and mag[i][j] >= mag[i][j + 1]:
                        mag_thin[i][j] = mag[i][j]
                if orient[i][j] == 1:
                    if mag[i][j] > mag[i - 1][j + 1] and mag[i][j] >= mag[i + 1][j - 1]:
                        mag_thin[i][j] = mag[i][j]
                if orient[i][j] == 2:
                    if mag[i][j] > mag[i - 1][j] and mag[i][j] >= mag[i + 1][j]:
                        mag_thin[i][j] = mag[i][j]
                if orient[i][j] == 3:
                    if mag[i][j] > mag[i - 1][j - 1] and mag[i][j] >= mag[i + 1][j + 1]:
                        mag_thin[i][j] = mag[i][j]

        # 4) Thresholding and edge linking

        result_binary = np.zeros(mag_thin.shape)

        tHigh = t_high
        tLow = t_low
        # forward scan
        for i in range(0, mag_thin.shape[0] - 1):  # rows
            for j in range(0, mag_thin.shape[1] - 1):  # columns
                if mag_thin[i][j] >= tHigh:
                    if mag_thin[i][j + 1] >= tLow:  # right
                        mag_thin[i][j + 1] = tHigh
                    if mag_thin[i + 1][j + 1] >= tLow:  # bottom right
                        mag_thin[i + 1][j + 1] = tHigh
                    if mag_thin[i + 1][j] >= tLow:  # bottom
                        mag_thin[i + 1][j] = tHigh
                    if mag_thin[i + 1][j - 1] >= tLow:  # bottom left
                        mag_thin[i + 1][j - 1] = tHigh

        # backwards scan
        for i in range(mag_thin.shape[0] - 2, 0, -1):  # rows
            for j in range(mag_thin.shape[1] - 2, 0, -1):  # columns
                if mag_thin[i][j] >= tHigh:
                    if mag_thin[i][j - 1] > tLow:  # left
                        mag_thin[i][j - 1] = tHigh
                    if mag_thin[i - 1][j - 1]:  # top left
                        mag_thin[i - 1][j - 1] = tHigh
                    if mag_thin[i - 1][j] > tLow:  # top
                        mag_thin[i - 1][j] = tHigh
                    if mag_thin[i - 1][j + 1] > tLow:  # top right
                        mag_thin[i - 1][j + 1] = tHigh

        # fill in result_binary
        for i in range(0, mag_thin.shape[0] - 1):  # rows
            for j in range(0, mag_thin.shape[1] - 1):  # columns
                if mag_thin[i][j] >= tHigh:
                    result_binary[i][j] = 255  # set to 255 for >= tHigh

        #cv2.imshow("mag_thin", result_binary)

        #img = cv2.imread('Images/minions.jpg', 0)
        #gnh = cv2.Canny(img, img,20, 30)

        img = Image.fromarray(result_binary)
        #img = img.convert('1')
        return img


    def drogEdgeDetectorFilter(self, img, ret_grad=False, pillow=True):

        if pillow:
            img = img.convert('L')
        open_cv_image = np.array(img)
        width, height = 0, 0
        if not pillow:
            height, width = img.shape

        else:
            width, height = img.size

        cv2.GaussianBlur(open_cv_image, (15, 15), 0)

        # Create sobel x and y matrix
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

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
        img = img.convert('L')
        if ret_grad:
            return Gx, Gy
        return img

    def medianFilter(self, img):

        img = img.convert('RGB')
        im = np.array(img)
        red = im[:, :, 0]
        green = im[:, :, 1]
        blue = im[:, :, 2]
        w = 1

        for i in range(w, im.shape[0] - w):
            for j in range(w, im.shape[1] - w):
                block_red = red[i - w:i + w + 1, j - w:j + w + 1]
                m_r = np.median(block_red)
                red[i][j] = int(m_r)

                block_green = green[i - w:i + w + 1, j - w:j + w + 1]
                m_g = np.median(block_green)
                green[i][j] = int(m_g)

                block_blue = blue[i - w:i + w + 1, j - w:j + w + 1]
                m_r = np.median(block_blue)
                blue[i][j] = int(m_r)

        im[:, :, 0] = red
        im[:, :, 1] = green
        im[:, :, 2] = blue
        img = Image.fromarray(im)
        img = img.convert('RGB')
        return img

    def harmonicMeanFilter(self, img):

        img = img.convert('RGB')
        px = img.load()
        width, height = img.size
        result = np.array(img)

        kernel_dim = 1  # (*2+1)

        # kernel 3x3
        for i in range(kernel_dim, height - kernel_dim):
            for j in range(kernel_dim, width - kernel_dim):
                sum_red, sum_green, sum_blue = (0, 0, 0)
                for n in range(i - kernel_dim, i + kernel_dim + 1):
                    for m in range(j - kernel_dim, j + kernel_dim + 1):
                        r, g, b = px[m, n]
                        if r == 0: r = 1
                        if g == 0: g = 1
                        if b == 0: b = 1
                        sum_red += 1/r
                        sum_green += 1/g
                        sum_blue += 1/b


                num = (kernel_dim * 2 + 1)**2
                result[i, j] = tuple([int(num/sum_red), int(num/sum_green), int(num/sum_blue)])

        img = Image.fromarray(result)
        return img

    def geometricMeanFilter(self, img):


        img = img.convert('RGB')
        px = img.load()
        width, height = img.size
        result = np.array(img)

        kernel_dim = 1 #(*2+1)

        # kernel 3x3
        for i in range(kernel_dim, height - kernel_dim):
            for j in range(kernel_dim, width - kernel_dim):
                prod_red, prod_green, prod_blue = (1, 1, 1)
                for n in range(i - kernel_dim, i + kernel_dim + 1):
                    for m in range(j - kernel_dim, j + kernel_dim + 1):
                        r, g, b = px[m, n]
                        prod_red *= r
                        prod_green *= g
                        prod_blue *= b

                div = (kernel_dim * 2 + 1)**2
                result[i, j] = tuple([prod_red ** (1 / div), prod_green ** (1 / div), prod_blue ** (1 / div)])

        img = Image.fromarray(result)
        return img
        


        '''
        img = img.convert('RGB')
        im = np.array(img)
        red = im[:, :, 0]
        green = im[:, :, 1]
        blue = im[:, :, 2]
        w = 2

        print(red)

        for i in range(w, im.shape[0] - w):
            for j in range(w, im.shape[1] - w):
                block_red = red[i - w:i + w + 1, j - w:j + w + 1]
                m_r = np.prod(block_red/255)
                print(m_r)
                red[i][j] = m_r

                block_green = green[i - w:i + w + 1, j - w:j + w + 1]
                m_g = np.prod(block_green)**(1/9)
                green[i][j] = m_g

                block_blue = blue[i - w:i + w + 1, j - w:j + w + 1]
                m_b = np.prod(block_blue)**(1/9)
                blue[i][j] = m_b

        print(red)

        im[:, :, 0] = red
        im[:, :, 1] = green
        im[:, :, 2] = blue
        img = Image.fromarray(im)
        #img = img.convert('RGB')
        return img

        '''


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

        mean_arithmetic = np.ones((3, 3))*(1/9)

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


def my_Normalize(img):

    # convert into range of [0,1]
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    output = (img.astype('float') - min_val) / (max_val - min_val)

    return output