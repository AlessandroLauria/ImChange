# picBloom
picBloom is a cross-platform desktop app focused on the image processing. It implements the operation listened below:
- Color Change:
  - Brightness
  - Contrast
  - Saturation
- Blur filters:
  - Arithmetic mean
  - Geometrich mean
  - Harmonic mean
  - Median
- Edge detection:
  - Canny
  - DroG
- SRM
- Transformation:
  - Translate x-y
  - Scaling
  - Rotation

# Installation on OsX
To install the complete application on Apple devices you have to download the .app file from the following link:
http://picbloom.altervista.org/home/index.html#about

After that, place the .app in your Application folder and double click on it

# Installation on Windows

# Installation on Linux

# To contribute:
Clone the repository and work in the version for your OS. 

To run the code you have to lunch the following command, by command line:
- python picBloom.py

If everything go right the main window will open, if not, install the dependecy required:
- pip install pillow
- pip install pyqt5
- pip install opencv-python
- pip install imutils
