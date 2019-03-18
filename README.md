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

# To contribute
Clone the repository and work in the version for your OS. 

To run the code you have to lunch the following command, by command line:
- python picBloom.py

If everything go right the main window will open, if not, install the dependecy required:
- pip install pillow
- pip install pyqt5
- pip install opencv-python
- pip install imutils

# Software structure
The software is splitted in many files. The task of any is described below:
- picBloom.py:
   This is the main window that handle the grafic intefarce of the application. Lauch this file with python (or python3) to start the application.
- Filters.py:
   A library that implements all filters used by the application. If you want to add one new filter you need to update this file and use the new filter in picBloom.py following the "work flow" for the user interface.
- MessageBox.py:
   Probabily it not need to be modified. It open a message box that allow the user to use the filters selected.
  
All images used by the application are placed in the 'Images' folder.

# Future Developements

- The application need of course an implementation of more filters. 
- Some bugs have to be fixed. 
- The SRM algorithm need an improvement.

