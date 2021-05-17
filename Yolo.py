import cv2
import matplotlib.pyplot as plt
import darknet
import utils
import os
import torch
from tkinter import *

root = Tk()
root.title('Yolov3_detect_object')
root.geometry("500x450")

path_Yolo = os.getcwd()

# Set the location and name of the cfg file
cfg_file = path_Yolo + '\\cfg\\yolov3.cfg'
# Set the location and name of the pre-trained weights file
weight_file = path_Yolo + '\\weights\\yolov3.weights'
# Set the location and name of the COCO object classes file
namesfile = path_Yolo + '\\data\\coco.names'
# Load the network architecture
m = darknet.Darknet(cfg_file)

# Load the pre-trained weights
m.load_weights(weight_file)
# Load the COCO object classes
class_names = utils.load_class_names(namesfile)

def clear():
    my_text.delete(1.0, END)

def get_text():
    my_label.config(text=my_text.get(1.0, END))
    i_path = my_text.get(1.0, END)
    print(type(i_path), end="///")
    print(i_path)
    peter = "D:/Picture/test.jpg"
    print(type(peter))
    i_path2= i_path.replace("\\","/")
    i_path3 = i_path2.replace("\n","")
    print(peter)
    image_detect(i_path3)


def image_detect(i_path):

    # Load the image

    print(i_path)

    img = cv2.imread(i_path)
    # Convert the image to RGB
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # We resize the image to the input width and height of the first layer of the network.    
    resized_image = cv2.resize(original_image, (m.width, m.height))

    # Set the NMS threshold
    nms_thresh = 0.6
    # Set the IOU thresholdectec
    iou_thresh = 0.4

    # Detect objects in the image
    boxes = utils.detect_objects(m, resized_image, iou_thresh, nms_thresh)

    # Print the objects found and the confidence level
    utils.print_objects(boxes, class_names)
    #Plot the image with bounding boxes and corresponding object class labels
    # Set the default figure size
    plt.rcParams['figure.figsize'] = [24.0, 14.0]
    utils.plot_boxes(original_image, boxes, class_names, plot_labels = True)

    img = original_image.copy()
    width = img.shape[1]
    height = img.shape[0]
        
    for i in range(len(boxes)):
        box = boxes[i]
        # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
        # of the bounding box relative to the size of the image. 
        x1 = int(np.around((box[0] - box[2]/2.0) * width))
        y1 = int(np.around((box[1] - box[3]/2.0) * height))
        x2 = int(np.around((box[0] + box[2]/2.0) * width))
        y2 = int(np.around((box[1] + box[3]/2.0) * height))
        
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))
            print("left top right bottom :", x1, y1, x2, y2)


my_label = Label(root, text="Enter the picture's path here")
my_label.pack(pady=1)

my_text = Text(root, width=100, height = 1)
my_text.pack(pady=20)
button_frame = Frame(root)
button_frame.pack()

clear_button = Button(button_frame, text="Clear path", command=clear)
clear_button.grid(row = 0, column=0)

get_text_button = Button(button_frame, text="Dectect object", command=get_text)
get_text_button.grid(row=0,column=1, padx=20)

my_label = Label(root, text='')
my_label.pack(pady=20)

root.mainloop()