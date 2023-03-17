import cv2 as cv
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
from mergeaudio import merge_advd

forward_path = os.path.abspath(os.path.dirname(os.getcwd()))

parser = argparse.ArgumentParser(description='videoname')
parser.add_argument('videoname', type=str)
parser.add_argument('outputname', type=str)
args = parser.parse_args()
videoname = args.videoname
outputname = args.outputname

process_frame=''   
cap = cv.VideoCapture(f"{forward_path}/data/origin_videos/{videoname}.mp4")
f = 1
fr = 51
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if f == fr:
        process_frame = frame
        break
    f+=1
edge = cv.Canny(process_frame,100,200)
# cv.imshow('im',process_frame)
cv.imwrite(f"{forward_path}/data/processed_data/origin_process_image/{outputname}.jpg",process_frame)
# cv.imshow('edge',edge)
cv.imwrite(f"{forward_path}/data/processed_data/canny_edge/{outputname}.jpg",edge)

# HSV image
imhsv = cv.cvtColor(process_frame, cv.COLOR_BGR2HSV)
color1 = np.array([0,0,0])
color2 = np.array([0,0,0])
x1 = 0
y1 = 0
x2 = 0
y2 = 0

def save_video(mask):
    cap = cv.VideoCapture(f"{forward_path}/data/origin_videos/{videoname}.mp4")
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(f"{forward_path}/data/output_videos/no_audio/{outputname}.mp4", fourcc, fps, (width, height))
    c = 1
    framerate = 52
    while cap.isOpened():
        ret, frame = cap.read()
        # print("frame:",c)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        if c==framerate:
            break
        elif c==framerate-1:
            c+=1
            frame = cv.addWeighted(frame,1,mask,1,0)
            cv.imwrite(f"{forward_path}/data/processed_data/processed_image/{outputname}.jpg",frame)
            out.write(frame)
            continue
            
        c+=1
        frame = cv.addWeighted(frame,1,mask,1,0)
        out.write(frame)
        if cv.waitKey(10) == ord('q'):
            break
    cap.release()
    out.release()
    merge_advd(outputname)

def find_cour():
    H_range = 10
    S_range = 20
    V_range = 20
    
    mask_1 = cv.inRange(imhsv, 
                        np.array([0 if color1[0]-H_range<0 else color1[0]-H_range
                                         ,0 if color1[1]-S_range<0 else color1[1]-S_range
                                         ,0 if color1[2]-V_range<0 else color1[2]-V_range]), 
                        np.array([180 if color1[0]+H_range>180 else color1[0]+H_range
                                         ,255 if color1[1]+S_range>255 else color1[1]+S_range
                                         ,255 if color1[2]+V_range>255 else color1[2]+V_range]))
    mask_2 = cv.inRange(imhsv, 
                        np.array([0 if color2[0]-H_range<0 else color2[0]-H_range
                                         ,0 if color2[1]-S_range<0 else color2[1]-S_range
                                         ,0 if color2[2]-V_range<0 else color2[2]-V_range]), 
                        np.array([180 if color2[0]+H_range>180 else color2[0]+H_range
                                         ,255 if color2[1]+S_range>255 else color2[1]+S_range
                                         ,255 if color2[2]+V_range>255 else color2[2]+V_range]))
    
    cv.imwrite(f"{forward_path}/data/processed_data/mask1/{outputname}.jpg",mask_1)
    cv.imwrite(f"{forward_path}/data/processed_data/mask2/{outputname}.jpg",mask_2)
    mask_merge = cv.bitwise_or(mask_1,mask_2)
    cv.imwrite(f"{forward_path}/data/processed_data/mask_merge/{outputname}.jpg",mask_merge)
    closing = cv.morphologyEx(mask_merge, cv.MORPH_CLOSE,  cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    # cv.imshow("Display closing", closing)
    cv.imwrite(f"{forward_path}/data/processed_data/closing/{outputname}.jpg",closing)
    
    dmask = cv.dilate(closing, None, iterations=10)
    finalmask = cv.erode(dmask, None, iterations=3)
    # cv.imshow("Display erode", emask)
    cv.imwrite(f"{forward_path}/data/processed_data/final_mask/{outputname}.jpg",finalmask)
    
    finaledge = cv.bitwise_and(edge,finalmask)
    # cv.imshow('edge && emask',edgemask)
    cv.imwrite(f"{forward_path}/data/processed_data/final_edge/{outputname}.jpg",finaledge)
    finaledge_3 = cv.merge([finaledge,finaledge,finaledge])
    # cv.imshow('cn3edgemask',cn3edgemask)
    
    save_video(finaledge_3)
                
    
def getpos(event,x,y,flags,param):
    global color1, color2, x1, y1, x2, y2
    if event == cv.EVENT_LBUTTONDOWN:
        # print("pos:",x,y)
        print("HSV value: ", imhsv[y,x])
        # print(type(imhsv[y,x]))
        if getpos.click == 0:
            x1 = x
            y1 = y
            color1 = np.copy(imhsv[y,x])
            getpos.click+=1
        elif getpos.click == 1:
            x2 = x
            y2 = y
            color2 = np.copy(imhsv[y,x])
            cv.destroyWindow("Choose 2 points")
            find_cour()

getpos.click=0

cv.imshow("Choose 2 points", process_frame)
cv.setMouseCallback("Choose 2 points",getpos)
cv.waitKey(0)
