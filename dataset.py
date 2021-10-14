import tensorflow as  tf 
import cv2 
import numpy as np
import os 
import pandas as pd 
import cv2

class VOCDataset(tf.keras.utils.Sequence):
    def __init__(self, csv_file , img_dir , label_dir , S = 7 , B = 2, C = 20 , batch_size = 16 , shuffle = True ,  augmenter = None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.augmenter = augmenter
        self.S = S 
        self.B = B
        self.C = C
        self.batch_size = batch_size 
        
        if self.batch_size > len(self.annotations) :
            raise ValueError("Batch size should not be greater than data size.")
        
        self.shuffle = shuffle 
        self._on_epoch_end()
        

    def __len__(self):
        return int(len(self.annotations)/ self.batch_size)
    
    def __getitem__(self , index):
        batch_imgs ,batch_labels = [] , []
        for ix in range(self.batch_size*index , self.batch_size*(index+1)):
            img , labels = self.__get_single_item__(ix)
            batch_imgs.append(img)
            batch_labels.append(labels)

        return np.array(batch_imgs).astype('float32') , np.array(batch_labels).astype('float32')

    def _on_epoch_end(self):
        if self.shuffle :
            self.annotations = self.annotations.sample(frac = 1)
            self.annotations = self.annotations.reset_index(drop = True)

    def __get_single_item__(self , index):
        label_path = os.path.join(self.label_dir ,  self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f :
            for label in f.readlines():
                class_label , x , y , w , h = [float(x) if float(x) != int(float(x)) else int(x) for x in  label.replace("\n" , "").split()]
                boxes.append([class_label , x , y , w , h])
        img_path = os.path.join(self.img_dir ,  self.annotations.iloc[index,0])
        image = cv2.imread(img_path)
        image = image/127.5 -1
        boxes = np.array(boxes)

        if  self.augmenter : 
            image , boxes = self.augmenter(images = [image])[0] , boxes
        
        label_matrix = np.zeros((self.S , self.S ,  self.C + 5*self.B))
        for box in boxes :
            class_label , x , y ,  w , h = list(box)
            class_label = int(class_label)
            i , j = int(self.S*x) , int(self.S*y)
            x_cell ,  y_cell = self.S*x - i , self.S*y - j
            w_cell , h_cell = w*self.S ,  h*self.S

            if label_matrix[j , i , 20] == 0 :
                label_matrix[j , i , 20 ] = 1
                label_matrix[j , i , 21:25] = np.array([x_cell , y_cell , w_cell , h_cell])
        return image , label_matrix   

    def _plot_data(self , index) :
        img_path = os.path.join(self.img_dir ,  self.annotations.iloc[index,0])
        img = cv2.imread(img_path)
        orig_img  = img.copy()
        img_h, img_w = img.shape[:2]

        label_path = os.path.join(self.label_dir ,  self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f :
            for label in f.readlines():
                class_label , x , y , w , h = [float(x) if float(x) != int(float(x)) else int(x) for x in  label.replace("\n" , "").split()]
                boxes.append([class_label , x , y , w , h])
                sxp , syp , exp , eyp = int(img_w*(x - w/2)) , int(img_h*(y - h/2)) , int(img_w*(x + w/2)) , int(img_h*(y + h/2))  
                img = cv2.rectangle(img , (sxp , syp) , (exp ,  eyp) , (255 ,0 ,0) , 2)
        return orig_img , img
        


    
if __name__ == "__main__":
    dataset = VOCDataset("archive/100examples.csv" , "archive/images" , "archive/labels" , batch_size = 4)
    print("batch_output :  " , dataset[0][0].shape ,  dataset[0][1].shape)
    while True :
        _ , res_img = dataset._plot_data(np.random.randint(0 , 50))
        cv2.imshow("WINDOW" , res_img)
        if cv2.waitKey() == ord('q') :
            cv2.destroyAllWindows()
            break  

        