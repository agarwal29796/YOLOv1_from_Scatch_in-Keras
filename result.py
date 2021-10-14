import tensorflow as tf 
import os 
import numpy as np 
import cv2 
from architecture import YOLOv1
from dataset import VOCDataset
from utills import (non_max_supression , mean_average_precision)


# cell_boxes shape : (1 , 1470)  =>  
def cell_box_to_bbox(cell_boxes):
    cell_boxes = np.reshape(cell_boxes ,  (1, 7 , 7 ,30))
    cell_boxes1 = cell_boxes[...,21:25]
    cell_boxes2 = cell_boxes[...,26:30]
    # selecting box with highest probability 
    best_box = np.argmax(np.stack([cell_boxes[...,20:21] , cell_boxes[...,25:26]] , axis = -1) ,  axis = -1)
    final_cellboxes = (1- best_box)*cell_boxes1 + best_box*cell_boxes2
    tmp_arr = np.arange(0 , 7).reshape(1, 7)
    cell_indices = np.repeat(tmp_arr , [7] ,axis = 0).reshape(1,  7 ,7 ,1) 
    x = (1/7)*(final_cellboxes[...,0:1] + cell_indices)    
    y = (1/7)*(final_cellboxes[...,1:2] + cell_indices.T)
    wh =  (1/7)*(final_cellboxes[...,2:4])
    ## it is with respect to image
    converted_boxes = np.concatenate((x,y,wh) , axis = -1)
    predicted_class = np.argmax(cell_boxes[...,:20] , axis = -1).reshape(1, 7,7,1)
    best_confidence = np.maximum(cell_boxes[...,20:21] , cell_boxes[...,25:26] )

    converted_prediction = np.concatenate((predicted_class , best_confidence, converted_boxes) , axis = -1)
    return converted_prediction

    # print(best_box)


model_path = "./TRAINING_LOGS/1/best/training"
model = tf.keras.models.load_model(model_path , compile = False)


dataset = VOCDataset("archive/100examples.csv" , "archive/images" , "archive/labels" , batch_size = 1 , shuffle = False)
for _ in range(10):
    idx =  np.random.randint(0, 20)
    orig_img , res_img = dataset._plot_data(idx)
    img_w , img_h = res_img.shape[:2]
    orig_img = cv2.resize(orig_img , (224 ,224))
    orig_img  = orig_img/127.5 - 1
    response_label = model.predict(orig_img.reshape(1 , 224 ,224 , 3))  
    converted_bboxes = cell_box_to_bbox(response_label)
    bboxes = []
    #box :  [class_no  ,  prob  , x , y , w , h]

    for i in range(7):
        for j in range(7):
            bboxes.append(converted_bboxes[0 , i ,  j , : ])
    bboxes = non_max_supression(bboxes , 0.4 , 0.5 )
    # mean_avg_pre = mean_average_precision()
    for box in bboxes:
        x , y , w , h =  box[2:]
        sxp , syp , exp , eyp = int(img_w*(x - w/2)) , int(img_h*(y - h/2)) , int(img_w*(x + w/2)) , int(img_h*(y + h/2))  
        res_img = cv2.rectangle(res_img , (sxp , syp) , (exp ,  eyp) , (255 ,255 ,0) , 2)
        
    cv2.imshow("WINDOW" , res_img)
    cv2.waitKey()

