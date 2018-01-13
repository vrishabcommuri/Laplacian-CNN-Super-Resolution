"""
###########################

python script to create gui

###########################
Citations

The opencv/tkinter framework courtesy of my TP Mentor Vasu Agrawal

github: https://github.com/VasuAgrawal/112-opencv-tutorial/blob/master/opencvTkinterTemplate.py

drawUnitedStatesFlag() and drawStar() borrowed from my lab 4 submission

getColor() and instructionsScreen() borrowed from tkinter docs

"""


import time
import sys
import os
import math
# Tkinter selector
if sys.version_info[0] < 3:
    from Tkinter import *
    import Tkinter as tk
else:
    from tkinter import *
    import tkinter as tk

import numpy as np
import cv2
from PIL import Image, ImageTk
from tkinter.colorchooser import *
from tkinter.colorchooser import askcolor

import scipy.io as sio
from network import upscale



def init(data):
    data.screen = 'introscreen'
    data.outframe = None
    data.displayframe = None
    data.stopped = True
    data.colorselector = None
    data.color = '#fbfff0'
    data.drawlist = []
    data.csize = 10
    data.shape = 'oval'
    data.command = None
    data.commandcounter = 0
    data.filecoordinates = []
    data.selected_coordinates = None
    data.selected_image = None
    data.timename = None
    data.shapelist = ['oval', 'flag', 'star']
    data.shapecounter = 0
    data.timercount = 0
    data.buttonwidth = data.width/16
    data.fontsize = 39
    data.offset = data.width/160
    data.frozen = True
    data.temp = 0
    data.loaded = False
    data.toggle = None
    data.factor = 128


def opencvToTk(frame):
    """Convert an opencv image to a tkinter image, to display in canvas."""
    frame = frame[:,:,:3]
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    tk_image = ImageTk.PhotoImage(image=pil_img)
    return tk_image


def mousePressed(event, data):
    print(event.x, event.y)
    if data.screen == 'introscreen':
        if event.x > data.width/2-data.buttonwidth/2\
         and event.x < data.width/2+data.buttonwidth/2\
         and event.y > 3*data.height/4-data.buttonwidth \
         and event.y < 3*data.height/4-data.buttonwidth/2:
            data.screen = 'intro'

        elif event.x > data.width/2-data.buttonwidth/2\
         and event.x < data.width/2+data.buttonwidth/2 \
         and event.y > 3*data.height/4 \
         and event.y < 3*data.height/4+data.buttonwidth/2:
            data.screen = 'load'

        elif event.x > data.width/2-data.buttonwidth/2\
         and event.y > 3*data.height/4+data.buttonwidth \
         and event.x < data.width/2+data.buttonwidth/2 \
         and event.y < 3*data.height/4+data.buttonwidth+data.buttonwidth/2:
        	data.screen = 'instructions'

    if data.screen == 'freeze':
    	if event.x > data.width/2-data.buttonwidth/2\
    	 and event.y > data.height-data.buttonwidth \
    	 and event.x < data.width/2+data.buttonwidth/2+2*data.offset \
    	 and event.y < data.height-data.buttonwidth/2:
    		data.timename = int(time.time())
    		print(type(data.outframe), data.outframe)
    		pix = data.outframe
    		pix_pil = Image.fromarray(pix)
    		pix_grey = pix_pil.convert('L')
    		pix_grey.save(str(data.timename)+'_.jpg')
    		with open(str(data.timename) + '.txt', 'w') as f1:
    			f1.write(str(data.drawlist))

    	elif event.x > 3*data.width/4-data.buttonwidth/2\
    	 and event.y > data.height-data.buttonwidth \
    	 and event.x < 3*data.width/4+data.buttonwidth/2+2*data.offset \
    	 and event.y < data.height-data.buttonwidth/2:
    		data.screen = 'intro'

    	elif event.x > 1*data.width/4-data.buttonwidth/2\
    	 and event.y > data.height-data.buttonwidth \
    	 and event.x < 1*data.width/4+data.buttonwidth/2+2*data.offset \
    	 and event.y < data.height-data.buttonwidth/2:
            data.screen = 'load'

    print(data.drawlist)

    if data.screen == 'freeze'\
     and event.y > data.height/2-data.height/2+data.buttonwidth \
     and event.y < data.height/2+data.height/2-data.buttonwidth \
     and event.x < data.width/2+data.width/2-data.buttonwidth \
     and event.x > data.width/2-data.width/2+data.buttonwidth: #and data.loaded == True:
    	print('here!')
    	if data.color!='None' or data.color!=None:
    		data.drawlist.append([(event.x, event.y), data.csize, data.color, data.shape])
    	drawFigures(data)
    #if data.screen == 'freeze' and event.y > data.height/2-600 and event.y < data.height/2+600 and data.loaded == False:
    #	data.drawlist.append([(event.x, event.y), data.csize, data.color, data.shape])
    #	drawFigures(data)
    if data.screen == 'load':
    	data.selected_coordinates = None
    	data.selected_coordinates = (event.x, event.y)

def keyPressed(event, data):
    if data.screen == 'load':
        if event.keysym == '1':
            data.factor = 32
        elif event.keysym == '2':
    	    data.factor = 64
        elif event.keysym == '3':
            data.factor = 128

    if event.keysym == 'q':
        data.root.destroy()
        
    if event.keysym == 'space' and data.screen == 'intro':
        data.outframe = data.frame.copy()
        # data.outframe = opencvToTk(data.outframe)
        data.displayframe = data.frame.copy()
        data.displayframe = opencvToTk(data.displayframe)
        data.screen = 'freeze'
        
    if event.keysym == 'Escape':
        data.screen = 'intro'
        data.drawlist = []
        data.temp = 0
        data.loaded = False

    if event.keysym == 'Up' and data.screen == 'freeze':
        if not data.csize > 500:
            data.csize += 10
            
    if event.keysym == 'Down' and data.screen == 'freeze':
        if not data.csize <= 0:
            data.csize -= 10
            
    if event.keysym == 'Return' and data.screen == 'load'\
     and data.selected_coordinates != None \
     and checkOverlap(data) == True:
        data.screen = 'freeze'
        upscaled = upscale(data.selected_image, data, data.factor)
        tk_img = ImageTk.PhotoImage(image=upscaled)
        print('here',data.selected_image)
        data.displayframe = tk_img
        data.drawlist = []
        if os.path.exists(str(data.selected_image)[:-5] + '.txt'):
            with open(str(data.selected_image)[:-5] + '.txt', 'r') as f2:
                s = ""
                for char in f2:
                    s += char
                data.drawlist = eval(s)
        print(data.drawlist)
        data.temp += 1
        data.loaded = True
        data.toggle = 'upsampled'
        data.canvas.create_rectangle(0,0,data.width,data.height, fill='black')
        resetFig(data)

     
    if event.keysym == 'l':
    	data.screen = 'load'
        
    if event.keysym == 's' and data.screen == 'freeze':
        data.timename = int(time.time())
        pix = data.outframe[200:1200, 0:1000]
        pix_pil = Image.fromarray(pix)
        pix_grey = pix_pil.convert('L')
        pix_grey.save(str(data.timename)+'_.jpg')

        #cv2.imwrite(str(data.timename)+'.jpg', x)
        # np.save(str(data.timename), x)
        # with open(str(data.timename) + '.npy.txt', 'w') as f1:
        #     f1.write(str(data.drawlist))
        
    if event.keysym == 'BackSpace' and data.screen == 'freeze':
        if len(data.drawlist)>0:
            data.drawlist.pop()
            resetFig(data)
            
    if (event.keysym == 'Right' or event.keysym == 'Left') and data.screen == 'freeze':
        if event.keysym =='Right':
        	data.shapecounter += 1
        if event.keysym == 'Left':
        	data.shapecounter -= 1
        data.shape = data.shapelist[data.shapecounter % 3]
        
    if event.keysym == 't':
        if data.loaded == True and data.screen == 'freeze':
            print(data.toggle)

            if data.toggle == 'downsampled':
                data.toggle = 'upsampled'
                low = Image.open(str(data.timename)+'down.jpg')
                low = low.resize((1400,1400), Image.BICUBIC)
                data.displayframe = ImageTk.PhotoImage(image=low)
                resetFig(data)

            elif data.toggle == 'upsampled':
                data.toggle = 'bicubic'
                ups = Image.open(str(data.timename)+'your_file.jpg')
                ups = ups.resize((1400,1400), Image.BICUBIC)
                data.displayframe = ImageTk.PhotoImage(image=ups)
                resetFig(data)

            elif data.toggle == 'bicubic':
                data.toggle = 'downsampled'
                bic = Image.open(str(data.timename)+'up_bicubic.jpg')
                bic = bic.resize((1400,1400), Image.BICUBIC)
                data.displayframe = ImageTk.PhotoImage(image=bic)
                resetFig(data)


def timerFired(data):
    data.timercount += 1
    data.timercount = data.timercount%20


def cameraFired(data):
    """Called whenever new camera frames are available.
    Camera frame is available in data.frame. You could, for example, blur the
    image, and then store that back in data. Then, in drawCamera, draw the
    blurred frame (or choose not to).
    """

    # For example, you can blur the image.
    # data.frame = cv2.GaussianBlur(data.frame, (11, 11), 0)
    # data.frame = cv2.cvtColor(data.frame, cv2.COLOR_BGR2HLS_FULL)
    data.frame = cv2.flip(data.frame,1)
   

def drawCamera(canvas, data):
    data.tk_image = opencvToTk(data.frame)
    if data.screen == 'intro':
        canvas.create_rectangle(0,0,data.width, data.height, fill='black',width=0)
        canvas.create_image(data.width/2, data.height/2, image=data.tk_image)
        canvas.create_text(data.width/2,
         data.buttonwidth/2, 
         fill='white', 
         font=("Purisa", int(data.width/80)),
         text="Click 'space' to capture!")

    elif data.screen == 'freeze' and data.temp == 0:
        print('this is causing it')
        canvas.create_rectangle(0,0,data.width, data.height, fill='black')
        canvas.create_image(data.width/2, data.height/2, image=data.displayframe)
        data.temp +=1


def drawFigures(data):
    if len(data.drawlist)>0:
        draw = data.drawlist[-1]

        if draw[3] == 'oval':
            data.canvas.create_oval((draw[0][0]-draw[1]/2, draw[0][1]-draw[1]/2), 
            (draw[0][0]+draw[1]/2, draw[0][1]+draw[1]/2), 
            fill=draw[2], 
            width=0)

        elif draw[3] == 'flag':
            drawUnitedStatesFlag(data.canvas, draw[0][0], draw[0][1], (3*draw[1]), (3*draw[1]))

        elif draw[3] == 'star':
            drawStar(data.canvas, draw[0][0], draw[0][1], draw[1], 5, draw[2])

    

def resetFig(data):
    data.canvas.create_image(data.width/2, data.height/2, image=data.displayframe)
    for draw in data.drawlist:
        if draw[3] == 'oval':
            data.canvas.create_oval((draw[0][0]-draw[1]/2, draw[0][1]-draw[1]/2), 
            (draw[0][0]+draw[1]/2, draw[0][1]+draw[1]/2), 
            fill=draw[2], 
            width=0)
        elif draw[3] == 'flag':
            drawUnitedStatesFlag(data.canvas, draw[0][0], draw[0][1], (3*draw[1]), (3*draw[1]))
        elif draw[3] == 'star':
            drawStar(data.canvas, draw[0][0], draw[0][1], draw[1], 5, draw[2])


def checkOverlap(data):
    if data.selected_coordinates != None:
        for coord in data.filecoordinates:
            if data.selected_coordinates[1] > coord[1]-40\
             and data.selected_coordinates[1] < coord[1]:
                return True
                
        return False

def loadImage(canvas, data):
    data.filecoordinates = []
    path = os.getcwd()
    files = os.listdir(path)
    counter = 0

    for fi in files:
        if str(fi[-4:]) == '.jpg': #or str(fi[-4:]) == '.npy':
            counter += 50
            canvas.create_text(10, 10+counter, 
            	fill = '#58d6fe', 
            	text = str(fi),
            	font=("Courier", data.fontsize), 
            	anchor=W)

            if [fi, 20+counter] not in data.filecoordinates:
                data.filecoordinates.append([fi, 20+counter])
        
    if data.selected_coordinates != None:
        for coord in data.filecoordinates:
            if data.selected_coordinates[1] > coord[1]-20 and data.selected_coordinates[1] < coord[1]:
                print(coord)
                canvas.create_text(10, 
                	coord[1]-data.offset, 
                	fill = 'purple', 
                	text = str(coord[0]),
                	font=("Courier", data.fontsize), 
                	anchor=W)
                data.selected_image = str(coord[0])
                
                
def drawUnitedStatesFlag(canvas, startx, starty, winWidth, winHeight):   
    # draw the stripes
    for i in range(13):
        start = winHeight*i/13
        end = winHeight*(i+1)/13
        # for even stripes make them red
        if i%2 == 0:
            canvas.create_rectangle(startx, starty+start, startx+winWidth, starty+end, fill = 'darkred', 
            width = 0)
        # for odd stripes make them white
        else:
            canvas.create_rectangle(startx,starty+start,startx+winWidth, starty+end, fill = 'white', 
            width = 0)
    # create a blue rectangle for the stars to go over
    canvas.create_rectangle(startx,starty,startx+winWidth/2.5,\
     starty+winHeight*7/13, fill = "navyblue", 
    width = 0)
    # define the width and height of the blue rectangle
    recWidth = winWidth/2.5
    recHeight = winHeight*7/13
    # draw a row of 6 stars with the right spacing
    for i in range(6):
        drawStar(canvas,startx+recWidth/12 + recWidth*i/6,starty+winHeight*1/26, \
        winHeight*4/(5*13), 5, "white")
    # draw a row of 6 stars with the right spacing
    for i in range(6):
        drawStar(canvas,startx+recWidth/12 + recWidth*i/6,\
        starty+winHeight/9 + winHeight*1/26, winHeight*4/(5*13), 5, "white")
    # draw a row of 5 stars with the right spacing
    for i in range(5):
        drawStar(canvas,startx+recWidth/6 + recWidth*i/6,\
        starty+winHeight/18 + winHeight*1/26, winHeight*4/(5*13), 5, "white")
    # draw a row of 6 stars with the right spacing
    for i in range(6):
        drawStar(canvas,startx+recWidth/12 + recWidth*i/6,\
        starty+winHeight*2/9 + winHeight*1/26, winHeight*4/(5*13), 5, "white")
    # draw a row of 5 stars with the right spacing
    for i in range(5):
        drawStar(canvas,startx+recWidth/6 + recWidth*i/6,\
        starty+winHeight*3/18 + winHeight*1/26, winHeight*4/(5*13), 5, "white")
    # draw a row of 6 stars with the right spacing
    for i in range(6):
        drawStar(canvas,startx+recWidth/12 + recWidth*i/6,\
        starty+winHeight*3/9 + winHeight*1/26, winHeight*4/(5*13), 5, "white")
    # draw a row of 5 stars with the right spacing
    for i in range(5):
        drawStar(canvas,startx+recWidth/6 + recWidth*i/6,\
        starty+winHeight*5/18 + winHeight*1/26, winHeight*4/(5*13), 5, "white")
    # draw a row of 6 stars with the right spacing
    for i in range(6):
        drawStar(canvas,startx+recWidth/12 + recWidth*i/6,\
        starty+winHeight*4/9 + winHeight*1/26, winHeight*4/(5*13), 5, "white")
    # draw a row of 5 stars with the right spacing
    for i in range(5):
        drawStar(canvas,startx+recWidth/6 + recWidth*i/6,\
        starty+winHeight*7/18 + winHeight*1/26, winHeight*4/(5*13), 5, "white")

              
                          
def drawStar(canvas, centerX, centerY, diameter, numPoints, color): 
    # there are 2n points that we need to graph   
    points = 2 * numPoints
    # create a list to hold tuples
    li = []
    # if there is an odd number of points
    if numPoints % 2 == 1:
        # for each point
         for i in range(points):
             # find the angle theta
             theta = (360*i)/points
             # if i is odd
             if i % 2 == 1:
                 # find the x coordinate
                 x = centerX + (diameter/2) * math.sin(math.radians(theta))
                 # find the y coordinate
                 y = centerY + (diameter/2) * math.cos(math.radians(theta))
                # if i is even
             else:
                 # find the x coordinate
                 x = centerX + ((3/8)*diameter/2) * \
                 math.sin(math.radians(theta))
                 # find the y coordinate
                 y = centerY + ((3/8)*diameter/2) * \
                 math.cos(math.radians(theta))
            # append to list
             li.append((x,y))
    # if there is an even number of points
    else:
        # for each point
        for i in range(points):
            # calculate theta
            theta = (360*i)/points
            # if i is even
            if i % 2 == 0:
                # find x and y
                x = centerX + (diameter/2) * math.sin(math.radians(theta))
                y = centerY + (diameter/2) * math.cos(math.radians(theta))
            # if i is odd
            else:
                # find the x and y points
                x = centerX + ((3/8)*diameter/2) * \
                math.sin(math.radians(theta))
                y = centerY + ((3/8)*diameter/2) * \
                math.cos(math.radians(theta))
            # append to list
            li.append((x,y))
    
    # create the polygon using the list of points        
    canvas.create_polygon(li, fill=color)


def redrawAll(canvas, data):
    if data.screen == 'intro':
        data.frozen = False
        data.temp = 0
    else:
        data.frozen = True
    if data.screen == 'introscreen':
        pic2 = cv2.imread('background.jpg',1)
        data.background = opencvToTk(pic2)
        canvas.create_rectangle(0,0, data.width, data.height, fill="black", width=0)
        canvas.create_image(data.width/2,data.height/2,image=data.background)
        canvas.create_text(data.width/2, data.height/4, 
        	text="Facial Image Super Resolution", 
        	fill='white', 
        	font=("Courier", data.fontsize*2))

        canvas.create_rectangle(data.width/2-data.buttonwidth/2,
            3*data.height/4, 
            data.width/2+data.buttonwidth/2,
            3*data.height/4+data.buttonwidth/2, 
            fill='#0099ff')

        canvas.create_text(data.width/2-data.buttonwidth/2, 
        	3*data.height/4, 
        	text="Load", 
        	fill='white', 
        	font=("Courier", data.fontsize), 
        	anchor=NW)

        canvas.create_rectangle(data.width/2-data.buttonwidth/2, 
        	3*data.height/4-data.buttonwidth, 
        	data.width/2+data.buttonwidth/2, 
        	3*data.height/4-data.buttonwidth/2, 
        	fill='#0099ff')

        canvas.create_text(data.width/2-data.buttonwidth/2+data.offset, 
        	3*data.height/4-data.buttonwidth, 
        	text="New", 
        	fill='white', 
        	font=("Courier", data.fontsize), 
        	anchor=NW) 

        canvas.create_rectangle(data.width/2-data.buttonwidth/2, 
        	3*data.height/4+data.buttonwidth, 
        	data.width/2+data.buttonwidth/2, 
        	3*data.height/4+data.buttonwidth+data.buttonwidth/2, 
        	fill='#0099ff')

        canvas.create_text(data.width/2-data.buttonwidth/2, 
        	3*data.height/4+data.buttonwidth, 
        	text="Help", 
        	fill='white', 
        	font=("Courier", data.fontsize), 
        	anchor=NW) 

    if data.screen == 'freeze' or data.screen == 'intro':
        drawCamera(canvas, data)
        
    if data.command != None:
        canvas.create_text(175,10, 
        	fill="black", 
        	text="Can only open color palette after capturing a frame! Hint: click \'space\'")
        data.commandcounter += 1
        if data.commandcounter == 60:
            data.commandcounter = 0
            data.command = None       
                            
    if data.screen == 'freeze':
        drawFigures(data)
        
        if data.shape == 'oval':
            data.canvas.create_rectangle(data.width/2-data.width/32,0, data.width/2+data.width/32, data.height/16,  fill='black', width=0)
            data.canvas.create_oval((data.width/2-data.width/32,0),(data.width/2+data.width/32, data.height/16), 
            fill=data.color, 
            width=0)

        elif data.shape == 'flag':
            data.canvas.create_rectangle(data.width/2-data.width/32,0, data.width/2+data.width/32, data.height/16, fill='black', width=0)
            drawUnitedStatesFlag(canvas, data.width/2-data.width/32,0, data.width/16, data.height/16)

        elif data.shape == 'star':
            data.canvas.create_rectangle(data.width/2-data.width/32,0, data.width/2+data.width/32, data.height/16, fill='black', width=0)
            drawStar(data.canvas, data.width/2, data.height/32, data.width/16, 5, data.color)

        canvas.create_rectangle(data.width/2-data.buttonwidth/2, 
        	data.height-data.buttonwidth,
        	data.width/2+data.buttonwidth/2+2*data.offset, 
        	data.height-data.buttonwidth/2, 
        	fill='#0099ff')

        canvas.create_text(data.width/2-data.buttonwidth/2+data.offset, 
        	data.height-data.buttonwidth, 
        	text="Save", 
        	fill='white', 
        	font=("Courier", data.fontsize), 
        	anchor=NW) 

        canvas.create_rectangle(3*data.width/4-data.buttonwidth/2, 
        	data.height-data.buttonwidth, 
        	3*data.width/4+data.buttonwidth/2+2*data.offset, 
        	data.height-data.buttonwidth/2, 
        	fill='#0099ff')

        canvas.create_text(3*data.width/4-data.buttonwidth/2+data.offset, 
        	data.height-data.buttonwidth, 
        	text="Back", 
        	fill='white', 
        	font=("Courier", data.fontsize), 
        	anchor=NW)   

        canvas.create_rectangle(1*data.width/4-data.buttonwidth/2, 
        	data.height-data.buttonwidth, 
        	1*data.width/4+data.buttonwidth/2+2*data.offset, 
        	data.height-data.buttonwidth/2, 
        	fill='#0099ff')

        canvas.create_text(1*data.width/4-data.buttonwidth/2+data.offset, 
        	data.height-data.buttonwidth, 
        	text="Load", 
        	fill='white', 
        	font=("Courier", data.fontsize), 
        	anchor=NW)

    if data.screen == 'load':
        canvas.create_rectangle((0,0), (data.width, data.height), fill='#2f292e')
        loadImage(canvas, data)

        
    if data.screen == 'instructions':
        canvas.create_rectangle((0,0), (data.width,data.height), fill='#2f292e')

        canvas.create_text(data.width/2, 
        	data.height/(data.offset+2), 
        	fill = '#58d6fe', 
        	text = "Instructions",
        	font=("Purisa", 30), 
        	anchor=N) 

        canvas.create_text(data.offset, 
        	2*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Click 'space' to capture an image",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)

        canvas.create_text(data.offset, 
        	3*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Mouseclick anywhere on a captured image to draw",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)

        canvas.create_text(data.offset, 
        	4*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Click or hold 'backspace' to erase the previous drawings",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)

        canvas.create_text(data.offset, 
        	5*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Use the 'left' and 'right' keys to change the brush",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)

        canvas.create_text(data.offset, 
        	6*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Use the color palette to change the brush color",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)

        canvas.create_text(data.offset, 
        	7*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Use the 'up' and 'down' keys to change the brush size",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)
        
        canvas.create_text(data.offset, 
        	8*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Use the 't' key to change toggle"\
        	+ " between downsampled, bicubic upscaled, and LAPSRN upscaled images",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)

        canvas.create_text(data.offset, 
        	9*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Click 's' to save a captured a image",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)

        canvas.create_text(data.offset, 
        	10*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Click 'l' to load an image",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)

        canvas.create_text(data.offset, 
        	11*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "Click 'escape' to return to the homepage",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)

        canvas.create_text(data.offset, 
        	12*data.height/(data.offset+4), 
        	fill = '#58d6fe', 
        	text = "temp",
        	font=("Purisa", int(data.width/80)), 
        	anchor=W)


def run(width=300, height=300):
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.camera_index = 0

    data.timer_delay = 100 # ms
    data.redraw_delay = 50 # ms
    
    # Initialize the webcams
    camera = cv2.VideoCapture(data.camera_index)
    data.camera = camera
    init(data)
    # Make tkinter window and canvas
    data.root = Tk()
    data.root.configure(background='black')
    
    data.fr = tk.Frame(data.root)
    data.fr.pack()
    
    def getColor(data = data):
        if data.screen == 'freeze':
            color = askcolor()
            if color[1] != None: 
                data.color = str(color[1])
        else:
            data.command = True
            
    def instructionsScreen(data=data):
        data.screen = 'instructions'
            

    button = tk.Button(data.fr, 
                       text="color palette", 
                       command=getColor)
    
    button1 = tk.Button(data.fr, 
                       text="instructions", 
                       command=instructionsScreen,)
    
    
    canvas = Canvas(data.root, width=data.width, height=data.height)
    data.canvas = canvas
    canvas.pack()
    


    # Basic bindings. Note that only timer events will redraw.
    data.root.bind('<B1-Motion>', lambda event: mousePressed(event, data))
    data.root.bind("<Button-1>", lambda event: mousePressed(event, data))
    data.root.bind("<Key>", lambda event: keyPressed(event, data))

    # Timer fired needs a wrapper. This is for periodic events.
    def timerFiredWrapper(data):
        # Ensuring that the code runs at roughly the right periodicity
        start = time.time()
        timerFired(data)
        end = time.time()
        diff_ms = (end - start) * 1000
        delay = int(max(data.timer_delay - diff_ms, 0))
        data.root.after(delay, lambda: timerFiredWrapper(data))

    # Wait a timer delay before beginning, to allow everything else to
    # initialize first.
    data.root.after(data.timer_delay, 
        lambda: timerFiredWrapper(data))

    def redrawAllWrapper(canvas, data):
        start = time.time()
        if data.screen == 'freeze':
        	button.pack(fill=NONE, side=tk.LEFT)
        	button1.pack(fill=NONE, side=tk.TOP)

        # Get the camera frame and get it processed.
        _, data.frame = data.camera.read()
        data.frame = cv2.resize(data.frame, (1600,1200))
        data.frame = data.frame[200:1400, 0:1200]

        data.frame = cv2.resize(data.frame, (1400, 1400))

        cameraFired(data)

        # Redrawing code
        if data.frozen == False:
            canvas.delete(ALL)
        redrawAll(canvas, data)

        # Calculate delay accordingly
        end = time.time()
        diff_ms = (end - start) * 1000

        # Have at least a 5ms delay between redraw. Ideally higher is better.
        delay = int(max(data.redraw_delay - diff_ms, 5))
        data.root.after(delay, lambda: redrawAllWrapper(canvas, data))
        

        
    # Start drawing immediately
    data.root.after(0, lambda: redrawAllWrapper(canvas, data))
    
    # Loop tkinter
    data.root.mainloop()

    # Once the loop is done, release the camera.
    print("Releasing camera!")
    data.camera.release()

if __name__ == "__main__":
    run(1600, 1600)



