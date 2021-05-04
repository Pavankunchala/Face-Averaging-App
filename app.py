import dlib
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import faceBlendCommon as fbc
import sys


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


@st.cache
def readImagePaths(path):
  # Create array of array of images.
  imagePaths = []
  # List all files in the directory and read points from text files one by one
  for filePath in sorted(os.listdir(path)):
    fileExt = os.path.splitext(filePath)[1]
    if fileExt in [".jpg", ".jpeg",'.png']:
      print(filePath)

      # Add to array of images
      imagePaths.append(os.path.join(path, filePath))

  return imagePaths

@st.cache
def averager(names):
    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
    
    # Read all images
    imagePaths = names

    if len(imagePaths) == 0:
        
        
        print('No images found with extension jpg or jpeg')
        sys.exit(0)

    #Read images and perform landmark detection.
    images = []
    allPoints = []
    
    for imagePath in imagePaths:
        im = cv2.imread(imagePath)
        if im is None:
            
            print("image:{} not read properly".format(imagePath))
        else:
            points = fbc.getLandmarks(faceDetector, landmarkDetector, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if len(points) > 0:
                allPoints.append(points)
          
                im = np.float32(im)/255.0
                images.append(im)
            else:
                print("Couldn't detect face landmarks")
          
          
    if len(images) == 0:
            
        print("No images found")
        sys.exit(0)
    
    w = 600
    h = 600

  # 8 Boundary points for Delaunay Triangulation
    boundaryPts = fbc.getEightBoundaryPoints(h, w)

    numImages = len(imagePaths)
    numLandmarks = len(allPoints[0])

  # Variables to store normalized images and points.
    imagesNorm = []
    pointsNorm = []

  # Initialize location of average points to 0s
    pointsAvg = np.zeros((numLandmarks, 2), dtype=np.float32)
    
    for i, img in enumerate(images):
        
    
        points = allPoints[i]
        points = np.array(points)

        img, points = fbc.normalizeImagesAndLandmarks((h, w), img, points)
        
        pointsAvg = pointsAvg + (points / (1.0*numImages))
        
        points = np.concatenate((points, boundaryPts), axis=0)
        
        pointsNorm.append(points)
        imagesNorm.append(img)
    pointsAvg = np.concatenate((pointsAvg, boundaryPts), axis=0)
    
    rect = (0, 0, w, h)
    dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)
    
    output = np.zeros((h, w, 3), dtype=np.float)
    
    for i in range(0, numImages):
        imWarp = fbc.warpImage(imagesNorm[i], pointsNorm[i], pointsAvg.tolist(), dt)
        
        output = output + imWarp
        
    output = output / (1.0*numImages)
    
    output = output[:,:,::-1]
    
    return output




st.title('Face Averaging')
st.text('Make sure you have a folder with atleast 3-6 images')
filename = st.text_input('Enter the Folder path:')

if filename == None:
    filename = 'images'


names = readImagePaths(filename)

st.subheader('Images in the folder')
st.image(names,width = 200,caption =['images in the folder'] * len(names))

st.subheader('Now we will create an image which is an **Average** of all these images')

output = averager(names)

st.image(output,caption ='Face Average')

st.markdown('''
          # About Author \n 
             Hey this is ** Pavan Kunchala ** I hope you like the application \n
             
            I am looking for ** Collabration **,** Freelancing ** and  ** Job opportunities ** in the field of ** Deep Learning ** and 
            ** Computer Vision **  if you are interested in my profile you can check out my resume from 
            [here](https://drive.google.com/file/d/1Mj5IWmkkKajl8oSAPYtAL_GXUTAOwbXz/view?usp=sharing)\n
            
            If you're interested in collabrating you can mail me at ** pavankunchalapk@gmail.com ** \n
            You can check out my ** Linkedin ** Profile from [here](https://www.linkedin.com/in/pavan-kumar-reddy-kunchala/) \n
            You can check out my ** Github ** Profile from [here](https://github.com/Pavankunchala) \n
            You can also check my technicals blogs in ** Medium ** from [here](https://pavankunchalapk.medium.com/) \n
            If you are feeling generous you can buy me a cup of ** coffee ** from [here](https://www.buymeacoffee.com/pavankunchala)
             
            ''')


