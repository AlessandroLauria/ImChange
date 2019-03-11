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


    def translate(self, img, x, y):
        img = img.convert('RGB')
        open_cv_image = np.array(img)

        red = open_cv_image[:, :, 0]
        green = open_cv_image[:, :, 1]
        blue = open_cv_image[:, :, 2]

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

        srm = SRM(image, n_sample)
        segmented = srm.run()
        segmented = np.array(segmented).astype(int)

        red = segmented[:, :, 0]
        green = segmented[:, :, 1]
        blue = segmented[:, :, 2]

        image[:, :, 0] = red
        image[:, :, 1] = green
        image[:, :, 2] = blue

        img = Image.fromarray(image)
        return img

    def cannyEdgeDetectorFilter(self, img, sigma=1, t_low=0.01, t_high=0.2):

        im = img.convert('L')
        img = np.array(im, dtype=float)

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


        img = Image.fromarray(result_binary)
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


    def arithmeticMeanFilter(self, img):

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