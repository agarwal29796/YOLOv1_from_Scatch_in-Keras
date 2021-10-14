import tensorflow as tf 
import numpy as np
from utills import intersection_over_union


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self , S = 7 , B = 2 , C = 20 , **kwargs ):
        # S =  split_size , B = num_blocks , C = num_classes
        super(YoloLoss , self).__init__()

        # non dimension reducible mse
        self.mse = tf.keras.losses.MeanSquaredError()

        self.S , self.B , self.C = S , B , C
        self.lambda_coord = 0.5
        self.lambda_noobj = 5

    def call(self , target , predictions):
        '''
        prediction : (batch_size ,  1470) , 1470 = S*S*(C  + B*5) , 5 -> ( obj_prob, x_coord , y_coord , w , h )
        target : (batch_size , S , S , C + 5) , B = 1
        '''

        # print("loss _err" , predictions.shape , target.shape)
        predictions = tf.reshape(predictions , (-1 , self.S , self.S , self.C + self.B*5))
        exists_box = target[... , 20:21] #shape should be (batch_size , S , S , 1)
        # print(predictions.shape ,  target.shape ,  exists_box.shape)

        iou_b1 = intersection_over_union(predictions[... , 21:25] , target[... , 21:25])  # return shape =  (batch_size , S , S , 4  ) 1 -> iou_value with target        
        iou_b2 = intersection_over_union(predictions[... , 26:30] , target[... , 21:25])        

        # print(iou_b1.shape)
        iou_maxes = tf.maximum(iou_b1 , iou_b2)
        # print(iou_maxes.shape)
        bestbox = tf.math.argmax(tf.stack([iou_b1 , iou_b2] , axis = -1) ,  axis = -1)
        #  iou_maxes (batch_size , S , S , 1 ) 1 -> max iou value between tow boxes   , bestboxes = (batch_size , S , S , 1) 1 -> ( 0 or 1 ) 0 mean  first box(21:25) and 1 mean second box(26 : 30)
        
        # print(bestbox.shape)
        exists_box = tf.cast(exists_box ,  tf.float32)
        bestbox = tf.cast(bestbox ,  tf.float32)

        # print(exists_box.shape  ,  target.shape)
        
        # print(exists_box.dtype ,  target.dtype ,  (1 - bestbox).dtype , predictions.dtype)
        box_predictions = exists_box * ( bestbox*predictions[... , 26 : 30] +  (1 - bestbox)*predictions[...,21: 25]) # shape (batch_size ,  S , S , 4)
        # print(box_predictions.shape)
        box_targets = exists_box  * target[...,21 : 25] # shape (batch_size ,  S , S , 4)

        # taking sqrt of width and height values of prediction  
        box_predictions_wh  = tf.sign(box_predictions[..., 2:4])*tf.sqrt(tf.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets_wh  = tf.sqrt(box_targets[... ,  2:4])

        box_loss1 = self.mse(tf.reshape(box_predictions[... ,0:2] ,(-1 , 2)) , tf.reshape(box_targets[...,0:2] ,(-1 , 2)))
        box_loss2 = self.mse(tf.reshape(box_predictions_wh ,(-1 , 2)) , tf.reshape(box_targets_wh ,(-1 , 2)))
        box_loss = box_loss1 + box_loss2

        #  OBJECT LOSS 
        pred_box = bestbox*predictions[..., 25:26] + (1 - bestbox)*predictions[..., 20:21]
        obj_loss = self.mse(exists_box*pred_box , exists_box*target[...,20:21])

        # NO OBJECT LOSS 
        # Interpretation 1 : where box exists -> reduce class prob loss for min iou box 
        # pred_box_nobj =  (1- best_box)*prediction[...,25:26] + best_box*predictions[...,20:21]
        # nobj_loss = self.mse(exists_box*pred_box_nobj ,  exists_box*target) #target prob is zero here 

        # Interpretation 2 : where box dont exists -> reduce class prob loss for both boxes 
        nobj_loss1 = self.mse((1 - exists_box)*predictions[..., 20:21] , (1 - exists_box)*target[...,20:21]) +  self.mse((1 - exists_box)*predictions[..., 25:26] , (1 - exists_box)*target[...,20:21])   # here also target prob is zero 

        #CLASS LOSS 
        class_loss = self.mse(exists_box*predictions[..., :20] , exists_box*target[...,:20])

        loss = self.lambda_coord*box_loss + obj_loss + self.lambda_noobj*nobj_loss1 +  class_loss
        # print("loss : " , loss)
        return loss         


if __name__ == "__main__":
    lossfn = YoloLoss()
    y_true =  np.random.rand(4 , 7 , 7 , 30).astype(np.float32)        
    y_pred =  np.random.rand(4 , 1470).astype(np.float32)
    print(y_pred.shape , y_true.shape)
    print(y_pred.dtype ,  y_true.dtype)
    print(lossfn(y_true , y_pred))




        
