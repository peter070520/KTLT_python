import cv2
import matplotlib.pyplot as plt
import darknet
import utils
import os
import torch
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
root = Tk()
root.title('Yolov3_detect_object')
root.geometry("1500x600")

filename = ''
path_Yolo = os.getcwd()
print(path_Yolo)
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
    """Function for clear button: Clear the picture path"""
    my_text.delete(1.0, END)

def get_text():
    """Function for detect button: Get the picture path, and start detecting object """
    i_path = my_text.get(1.0, END)
    print(type(i_path), end="///")
    print(i_path)
    i_path2= i_path.replace("\\","/")
    i_path3 = i_path2.replace("\n","")
    image_detect(i_path3)

def browseFiles():
    """Function for browse file button, browse the picture"""
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("image",
                                                        "*.jpg*"),
                                                       ("all files",
                                                        "*.*")))
    path = ''
    path = path + filename

    global my_image
    my_image = ImageTk.PhotoImage(Image.open(filename),0)
    my_image_label = Label(image=my_image).pack()
    # Change label contents
    my_text.insert(1.0, path)

def image_detect(i_path):
    """Starting detect object"""
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


    # Print the objects found and the confidence level
    utils.print_objects(boxes, class_names)
    #Plot the image with bounding boxes and corresponding object class labels
    # Set the default figure size
    plt.rcParams['figure.figsize'] = [24.0, 14.0]
    utils.plot_boxes(original_image, boxes, class_names, plot_labels = True)

my_label = Label(root, text="Enter the picture's path here")
my_label.pack(pady=1)

open_file = Frame(root)
open_file.pack()

my_text = Text(open_file, width=100, height = 1)
my_text.grid(row = 0, column=0)

button_explore = Button(open_file,
                        text = "Browse Files",
                        command = browseFiles)
button_explore.grid(row=0,column=1, padx=1)

button_frame = Frame(root)
button_frame.pack()

clear_button = Button(button_frame, text="Clear path", command=clear)
clear_button.grid(row = 0, column=0)

get_text_button = Button(button_frame, text="Dectect object", command=get_text)
get_text_button.grid(row=0,column=1, padx=20)


root.mainloop()