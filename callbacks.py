import tensorflow as tf 
import numpy as np


LR_SCHEDULE = {1: 0.05 , 10 : 0.005 , 30 : 0.001 , 100: 0.00001 , 200 : 0.0001}
filepath = "model_{epoch:02d}_{val_loss:.2f}"
EXP_NUMBER = 1
checkpoint_path = "./TRAINING_LOGS/" +  str(EXP_NUMBER) 





# We can write custom callbacks
class EarlyStopping(tf.keras.callbacks.Callback):
    '''
    Stop training when loss stop decreasing in next 'patience' epochs 
    '''
    def __init__(self, patience = 0):
        super(EarlyStopping ,  self).__init__()
        self.patience = patience 
        self.last_best_loss = np.Inf
        
    def on_train_begin(self , logs = None):
        self.waited = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch , logs=None):
        epoch_loss = logs['loss']
        if epoch_loss < self.last_best_loss:
            self.last_best_loss =  epoch_loss
            self.waited = 0 
        else:
            self.waited += 1 
            if self.waited > self.patience :
                self.stopped_epoch = epoch
                self.model.stop_training = True
    
    def on_train_end(self, logs = None):
        if self.stopped_epoch > 0 :
            print("\n\nEpoch %05d Early Stopping.\n\n" %(self.stopped_epoch ))



class LearningRateScheduler(tf.keras.callbacks.Callback):
    '''
    Writing a custom lr scheduler using callbacks
    '''
    def __init__(self ,  schedule , use_deafault_lr = False):
        super(LearningRateScheduler ,  self).__init__()
        self.schedule = schedule 
        self.use_deafault_lr = use_deafault_lr  

    def on_train_begin(self, logs = None):
        if not hasattr(self.model.optimizer , "lr"):
            raise ValueError("optimzer must have a lr attribute.")
        default_lr =  float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if self.use_deafault_lr : self.lr = default_lr

    def on_epoch_begin(self , epoch , logs = None):
        if self.use_deafault_lr :  return    

        all_keys = list(self.schedule.keys())
        final_key = int(all_keys[0])
        
        for key in  all_keys:
            if int(key) <= (epoch + 1) : final_key = key 
            else : break 
          
        self.lr = self.schedule[final_key]
        if final_key ==  epoch + 1: print("\nSetting Learning Rate to " , self.lr ," on epoch number : " ,epoch + 1 ,"\n")
        tf.keras.backend.set_value(self.model.optimizer.lr , self.lr)


# We can write custom callbacks
class ModelSave(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_path ,  saving_freq = 1 , saving_best = True):
        super(ModelSave ,  self).__init__()
        self.last_best_train_loss = np.Inf
        self.last_best_val_loss = np.Inf
        self.saving_freq = saving_freq
        self.saving_best  = saving_best
        self.checkpoint_path = checkpoint_path
        self.last_epoch = None

    def on_epoch_end(self, epoch , logs=None):
        self.last_epoch = epoch 
        current_train_loss =  min(self.last_best_train_loss , logs['loss'])
        current_val_loss =  min(self.last_best_val_loss , logs['val_loss'])
        if current_train_loss < self.last_best_train_loss and self.saving_best : 
            self.model.save(self.checkpoint_path + "/best/training" ,  save_format = 'tf')
            self.last_best_train_loss =  current_train_loss

        if current_val_loss < self.last_best_val_loss and self.saving_best : 
            self.model.save(self.checkpoint_path + "/best/validation" ,  save_format = 'tf')
            self.last_best_val_loss =  current_val_loss
        

        if epoch%self.saving_freq == 0:
            self.model.save(self.checkpoint_path + "/all_ckpts/" +  str(epoch + 1) ,  save_format = 'tf')

    def on_train_end(self , logs = None ):
        if self.last_epoch == None: return 
        self.model.save(self.checkpoint_path + "/all_ckpts/" +  str(self.last_epoch + 1) ,  save_format = 'tf')





# best_score_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path +"/best/"+filepath  ,monitor='val_loss', verbose=1,save_best_only=True, model='auto',period=1)

# all_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/all_ckpts/"+filepath, monitor='val_loss', save_weights_only = True ,verbose=0)

model_save_checkpoint =   ModelSave(checkpoint_path ,  saving_freq  = 5)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=checkpoint_path+"/logs/" )

early_stopping_callback =  EarlyStopping(patience  =  100)

lr_schedule_callback = LearningRateScheduler(LR_SCHEDULE )

