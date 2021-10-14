import tensorflow as tf
import os 
import numpy as np 
import imgaug.augmenters as iaa
from architecture import YOLOv1
from dataset import VOCDataset
from utills import (intersection_over_union , non_max_supression , mean_average_precision)
from callbacks import(early_stopping_callback , lr_schedule_callback ,  model_save_checkpoint , tensorboard_callback )
# from utills import ( cellboxes_to_boxes , get_bboxes , load_checkpoints , save_checkpoints )
from loss import YoloLoss

print("Tensorflow version : " ,  tf.__version__)
tf.random.set_seed(1234)

EXP_NUMBER = 1
IMG_WIDTH = IMG_HEIGHT = 224
LEARNING_RATE = 2e-5   
BATCH_SIZE = 4
WEIGHT_DECAY = 0 
EPOCHS =  200
NUM_WORKERS = 2 
MAX_QUEUE_SIZE = 60  
INITIAL_EPOCH = 0
IMG_DIR = "archive/images"
LABEL_DIR = "archive/labels"
CALLBACKS = [model_save_checkpoint , tensorboard_callback , lr_schedule_callback , early_stopping_callback]
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE) 
LOAD_PREV_MODEL = True
CHECKPOINT_PATH = "./TRAINING_LOGS/" +  str(EXP_NUMBER) 

# CREATING FOLDERS FOR CHECKPOINTS 
for homie_path in ["/best/" , "/all_ckpts/" , "/logs/"]:
    if not os.path.exists(CHECKPOINT_PATH + homie_path):
        os.makedirs(CHECKPOINT_PATH + homie_path , mode = 0o777)


sometimes = lambda aug: iaa.Sometimes(0.3, aug)

transforms = iaa.Sequential([
    iaa.Resize({"height" : IMG_HEIGHT , "width" : IMG_WIDTH}) , 
        sometimes(iaa.GaussianBlur(sigma=(1.5,2.5)))
  ])


def lastest_model_checkpoint(start_from_validation_best = False , start_from_training_best = False):
    if start_from_training_best :  
        return CHECKPOINT_PATH + "/best/training" 
    if start_from_validation_best:
        return  CHECKPOINT_PATH + "/best/validation"

    checkpoints = [CHECKPOINT_PATH + "/all_ckpts/" + name for name in os.listdir(CHECKPOINT_PATH +  "/all_ckpts/")]
    if len(checkpoints)  != 0:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        global INITIAL_EPOCH 
        INITIAL_EPOCH = int(latest_checkpoint.split('/')[-1]) 
        return latest_checkpoint

    print("\n\nno checkpoint found , to load existing model. :\n\n")
    return None



def main():
    model_config = {"split_size" : 7 , "num_boxes" : 2, "num_classes" : 20}
    last_checkpoint = lastest_model_checkpoint()
    model = YOLOv1(**model_config)

    ## setting models input shape
    aa = model(np.ones((4, IMG_HEIGHT , IMG_WIDTH , 3)))

    if LOAD_PREV_MODEL and last_checkpoint is not None : 
        print("\n\nLoading model from  :  ",last_checkpoint , "\n\n") 
        model = tf.keras.models.load_model(last_checkpoint , custom_objects = {'YOLOv1': YOLOv1 , 'YoloLoss': YoloLoss()})
    else: 
        print("\n\n Creatign a new model with fresh weights .\n\n")
        model.compile(optimizer = OPTIMIZER , loss = YoloLoss())

    
    train_dataset = VOCDataset("archive/100examples.csv" ,  IMG_DIR , LABEL_DIR ,batch_size= BATCH_SIZE, augmenter =  transforms)
    val_dataset = VOCDataset("archive/8examples.csv" ,  IMG_DIR , LABEL_DIR , batch_size=BATCH_SIZE, augmenter =  transforms)

    model.fit(x = train_dataset, validation_data = val_dataset ,initial_epoch = INITIAL_EPOCH , epochs = EPOCHS , callbacks = CALLBACKS )

if __name__ == "__main__":
    main()










