import numpy as np 
import tensorflow as tf 
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys 
import cv2 
import  matplotlib.pyplot as plt
from tensorflow.keras.layers import * 

print(tf.__version__)


#Tuple : (kernel size ,  no of kernels , stride , ? padding = same)
# List : [*tuples , repeat_count]
# "M" : maxpool layer : kernel : 2  ,  stride = 2 , padding = same
arch_config =  [
    (7 , 640 , 2), 
    "M" ,  
    (3, 192 , 1) , 
    "M" , 
    (1, 128 , 1) ,
    (3, 256 , 1) ,
    (1, 256 , 1) ,
    (3  ,512 , 1) ,
    "M" ,
    [(1 ,256 , 1) , (3 , 512 , 1) , 4], 
    (1 , 512 , 1), 
    (3 , 1024 , 1), 
    "M" ,
    [(1 , 512, 1) , (3 , 1024 , 1) , 2], 
    (3 , 1024, 1) ,
    (3 , 1024 , 2) ,
    (3 , 1024 , 1), 
    (3 , 1024, 1) 
]

class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size , kernel_count , stride = 1, padding = 'same'):
        super(CNNBlock , self).__init__()
        self.conv = tf.keras.layers.Conv2D( kernel_count , kernel_size , strides = stride , padding = padding , use_bias = False)
        self.batchnorm  =  BatchNormalization()
        self.leakyRelu =  LeakyReLU(alpha = 0.1)

    def call(self, x):
        return self.leakyRelu(self.batchnorm(self.conv(x)))


class YOLOv1(tf.keras.Model):
    def __init__(self, in_channels = 3 , **kwargs):
        super(YOLOv1 ,  self).__init__()
        self.arch_config =  arch_config 
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.arch_config)
        self.fcs = self._create_fcs(**kwargs)


    def call(self, x , training = False):
        # We can use optional training parameter to  set specific layer ( for ex : setting dropout onlfy for training . )        

        for layer in self.darknet :
            x = layer(x)  
        return self.fcs(Flatten()(x))

    def _create_conv_layers(self, architecture):
        layers = []
        for ly in architecture:
            if type(ly) == tuple :
                layers.append(CNNBlock(ly[0] ,  ly[1] ,  stride = ly[2]))
            elif type(ly) == list :
                for rep in range(ly[-1]):
                    for lly in ly[:-1] :
                        layers.append(CNNBlock(lly[0] , lly[1] , stride =  lly[2]))
            elif type(ly) == str:
                layers.append(MaxPool2D( pool_size = 2 , strides = 2 , padding = "same"))
        return layers

    def _create_fcs(self, split_size, num_boxes,  num_classes ):
        S ,  B , C = split_size , num_boxes ,  num_classes
        return tf.keras.Sequential([
            Flatten(), 
            Dense(496) ,  # in paper it is 4096
            LeakyReLU(alpha = 0.1), 
            Dense(S*S*( C + B*5)) 
        ])

    def summary1(self):
        inp = Input(shape = (224 , 224, 3))
        out = self.call(inp)
        model = tf.keras.models.Model(inputs = inp , outputs = out)
        print(model.summary())


final_arch_config = {"split_size" : 7 , "num_boxes" : 2, "num_classes" : 20}


if __name__ == "__main__":    
    model = YOLOv1(**final_arch_config)
    aa = model(np.ones((4, 224 , 224 , 3)))
    # model.summary1()
    model.compile(optimizer = tf.keras.optimizers.Adam() ,  loss = 'mse')
    tf.saved_model.save(model ,"model_save_check")
    # model.save("model_save_check")
    print("saved successfully.")
    # new_model = YOLOv1(**final_arch_config)
    new_model = tf.keras.models.load_model("model_save_check" , custom_objects =  {'YOLOv1' : YOLOv1})
    print(new_model(np.ones((4, 224 , 224 , 3))).shape)
    print(new_model.optimizer)





 

