
#  IOU CALCULATION
# input shape :  (... ,  x , y , w, h) x :[0 , 1]  , y : [0 , 1] , w : [0, ) h : [0, :)
# For two boxes [x1 , y1 , w1 , h1] , [x2 , y2 , w2 , h2]  where (x , y ) are init points  
# Intersection region  :  ( maximum\(x1 , x2) , maximum\(y1 , y2) ) to  ( minimum(x1 + w1 ,  x2 + w2) ,  minimum(y1 + h1 , y2 + h2))  

import numpy as np
import tensorflow as tf 
from collections import Counter

# boxes shape : (batch_shape , S , S , 4) 4 : [x , y , w,  h] where : (x , y: middle point,) 
def intersection_over_union(boxes_pred , boxes_label):
    box1_x1 = boxes_pred[...,0:1] - boxes_pred[...,2:3]/2
    box1_y1 = boxes_pred[...,1:2] - boxes_pred[...,3:4]/2
    box1_x2 = boxes_pred[...,0:1] + boxes_pred[...,2:3]/2
    box1_y2 = boxes_pred[...,1:2] + boxes_pred[...,3:4]/2

    box2_x1 = boxes_label[...,0:1] - boxes_label[...,2:3]/2
    box2_y1 = boxes_label[...,1:2] - boxes_label[...,3:4]/2
    box2_x2 = boxes_label[...,0:1] + boxes_label[...,2:3]/2
    box2_y2 = boxes_label[...,1:2] + boxes_label[...,3:4]/2

    x1 = tf.maximum(box1_x1 , box2_x1)
    y1 = tf.maximum(box1_y1 , box2_y1)
    x2 = tf.minimum(box1_x2 , box2_x2)
    y2 = tf.minimum(box1_y2 , box2_y2)

    intersection = tf.clip_by_value(x2 - x1 , 0 , tf.float32.max)*tf.clip_by_value(y2 - y1 ,0 , tf.float32.max)
    
    box1_area = tf.abs((box1_x2 - box1_x1)*(box1_y2 - box1_y1))
    box2_area = tf.abs((box2_x2 - box2_x1)*(box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area  - intersection + 1e-6)





# before nms try to discard boxes with lowest probability 

#  while bounding box : 
    # select largest prob box 
    # discard all other boxes which have iou > threshold with selected box 

# boxes : [ [ box_class , prob_bouding_box , x , y ,  w , h ] ]
def non_max_supression(boxes ,  min_prob_threshold , iou_threshold ,   box_format = "mid_point") :
    assert(type(boxes) == list)
    boxes = [box for box in boxes if box[1] >  min_prob_threshold]
    boxes = sorted(boxes , key = lambda x : x[1] , reverse=True)
    final_boxes = []
    while boxes :  
        choosen_box = boxes.pop(0)
        
        # boxes = [box  for box in boxes if box[0] != choosen_box[0]] 
        tmp_boxes = []
        for box in boxes:
            ii =  intersection_over_union(tf.constant(choosen_box[2:]),tf.constant(box[2:])).numpy()
            if box[0] != choosen_box[0] or ii < iou_threshold :  tmp_boxes.append(box) 
        boxes = tmp_boxes
        final_boxes.append(choosen_box)
    return final_boxes


def mean_average_precision(pred_boxes ,  true_boxes , iou_threshold= 0.5 , box_format="mid_point" ,num_classes = 20):
    # pred boxes : list with all boxes [img_idx , class_pred , prob_score , x , y,  w , h]
    average_precisions = []
    epsilon = 1e-6 
    for c in range(num_classes):
        detections = []
        ground_truths = []
        for detection in pred_boxes:
            if detection[1] == c : 
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1] == c :
                ground_truths.append(true_box)
        # img 0 has 3 boxes 
        # img 1 has 5 boxes 
        # amount boxes = {0 : 3 , 1 : 5}
        amount_bboxes = dict(Counter([gt[0] for gt in ground_truths]))
        # print(amount_bboxes)
        for key , val in amount_bboxes.items():
            amount_bboxes[key] = [0]*val

        detections.sort(key= lambda x : x[2] , reverse=True)
        TP =  [0]*len(detections)
        FP =  [0]*len(detections)
        total_true_boxes =  len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [box for box in ground_truths if box[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0 
            for idx , gt in enumerate(ground_truth_img):
                iou  =  intersection_over_union(tf.constant(detection[3:]) , tf.constant(gt[3:]))
                if iou > best_iou:
                    best_iou = iou  
                    best_gt_idx = idx
            if best_iou > iou_threshold: 
                if amount_bboxes[detection[0]][best_gt_idx] == 0 :
                    TP[detection_idx] = 1 
                    amount_bboxes[detection[0]][best_gt_idx] = 1 
                else:
                    FP[detection_idx] = 1
            else :
                FP[detection_idx] = 1
            
        # [1 , 1, 0 , 1, 0] -> [1, 2, ,2, 3, 3]
        TP_cumsum =  np.cumsum(TP).astype('float32') 
        FP_cumsum =  np.cumsum(FP) .astype('float32')

        recalls = np.divide(TP_cumsum , (total_true_boxes + epsilon))
        precisions = np.divide(TP_cumsum , (TP_cumsum + FP_cumsum + epsilon))

        # precision is y axis  , recall is x axis 
        precisions = np.concatenate((np.array([1]) ,  precisions))
        recalls = np.concatenate((np.array([0]) ,  recalls))

        average_precisions.append(np.trapz(precisions ,  recalls))
    return sum(average_precisions)/ len(average_precisions)









