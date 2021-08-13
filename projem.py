# -*- coding: utf-8 -*-
"""
Created on Thu May 20 01:41:11 2021

@author: elif
"""

from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import *

from PyQt5.uic import *
from PyQt5.Qt import QApplication, QUrl, QDesktopServices
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score
import random
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from glob import glob
import tensorflow.keras
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from skimage import io,color
import itertools
from tensorflow.keras import applications
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
#import scikitplot.metrics as splt
from scipy import interp

#from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder

from glob import glob
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import applications
from tensorflow.keras.layers import Concatenate
import tensorflow.keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout,BatchNormalization

class window(QMainWindow,QDialog):
    dosyamiz = None
    def __init__(self):
        super(window, self).__init__()
        loadUi("tasarim.ui", self)
        self.comboBox.addItem("vgg16")
        self.comboBox.addItem("Inception-V3")
        self.comboBox.addItem("Vhist")
        self.comboBox.addItem("vgg19")
        
        
        self.comboBox_2.addItem("vgg16")
        self.comboBox_2.addItem("Inception-V3")
        self.comboBox_2.addItem("Vhist")
        self.comboBox_2.addItem("vgg19")
        
        
        self.comboBox_3.addItem("Adam")
        self.comboBox_3.addItem("SGD")
        self.comboBox_3.addItem("RMSprop")
        
        self.comboBox_4.addItem("0.1")
        self.comboBox_4.addItem("0.2")
        self.comboBox_4.addItem("0.3")
       
        self.comboBox_5.addItem("16")
        self.comboBox_5.addItem("32")
        
       
        self.comboBox_6.addItem("5")
        self.comboBox_6.addItem("10")
        self.comboBox_6.addItem("20")
        self.comboBox_6.addItem("30")
        
       
        self.comboBox_7.addItem("0.01")
        self.comboBox_7.addItem("0.001")
        self.comboBox_7.addItem("0.0001")
        
        self.pushButton_5.clicked.connect(self.klasor)  
        self.pushButton.clicked.connect(self.islemler) 
        self.pushButton_2.clicked.connect(self.resimsec) 
        self.pushButton_3.clicked.connect(self.aa) 
        
    def klasor(self):
        file = str(QFileDialog.getExistingDirectory(self, "klasörü seç"))
        self.path=file.replace("C:/Users/elif/Desktop/Notlar/4.Sınıf/yapayzeka/","./")+"/"
        self.directories=os.listdir(self.path)  
    def resimsec(self):
        dosyaadi = QFileDialog.getOpenFileName()
        ilk , self.son = os.path.split(dosyaadi[0])
        print(self.son)
        self.image = cv2.imread(self.son)
        self.img = cv2.resize(self.image, (224,224))
        filename = 'savedImage.jpg'
        cv2.imwrite(filename, self.img)        
        self.pixmap = QPixmap("./"+filename)
        self.label_13.setPixmap(self.pixmap)
        #self.label_12.setText(self.son)
        
    def vgg16(self):
        
        IMAGE_SIZE = [224, 224]
        vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        for layer in vgg.layers:
            layer.trainable = False
        x = Flatten()(vgg.output)
        prediction = Dense(2, activation='softmax')(x)
        self.model = Model(inputs=vgg.input, outputs=prediction)
    def vgg19(self):
        IMAGE_SIZE = [224, 224]
        vgg19 = applications.VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        for layer in vgg19.layers:
            layer.trainable = False
        x = Flatten()(vgg19.output)
        prediction = Dense(2, activation='softmax')(x)
        self.model = Model(inputs=vgg19.input, outputs=prediction)
        
   
    def inceptionv3(self):
        inceptionv3 = applications.InceptionV3(input_shape=(224,224,3),  weights='imagenet', include_top=False)
        for layer in inceptionv3.layers:
            layer.trainable = False
        x = Flatten()(inceptionv3.output)
        prediction = Dense(2, activation='softmax')(x)
        self.model = Model(inputs=inceptionv3.input, outputs=prediction) 
        
    def vgg(self): 
         model1 = tensorflow.keras.applications.VGG16(input_shape=[244,224]+ [3],weights = 'imagenet', include_top = False)
         model2 = tensorflow.keras.applications.InceptionV3(input_shape=[244,224] + [3], weights='imagenet', include_top=False)
         for layer in model1.layers:
            layer.trainable = False
         for layer in model2.layers:
            layer.trainable = False  
         model1_out=model1.output
         model1_final=GlobalAveragePooling2D()(model1_out)
         model2_out=model2.output
         model2_final=GlobalAveragePooling2D()(model2_out)
         merged_model=Concatenate()([model1_final, model2_final])
         final_model=Dense(512,activation='relu')(merged_model)
         final_model=Dense(2,activation='sigmoid')(final_model)
         self.model = Model(inputs=[model1.input, model2.input], outputs=final_model)
         

    def islemler(self):
        validation_split = self.comboBox_4.currentText()
        batch_size = self.comboBox_5.currentText()
        nb_epochs = self.comboBox_6.currentText()
        learning_rate = self.comboBox_7.currentText()
        
        if self.comboBox.currentText()=="vgg16":
                 self.modeladi="vgg16.h5"
                 self.label_5.setText(str('VGG GRAFİK SONUÇLARI '))
                 self.vgg16()
        if self.comboBox.currentText()=="Inception-V3":
               self.modeladi="Inception-V3.h5"
               self.label_5.setText(str('Inception-V3 GRAFİK SONUÇLARI '))
               self.inceptionv3()
        if self.comboBox.currentText()=="Vhist":
               self.modeladi="Vhist.h5"
               self.label_5.setText(str('Vhist GRAFİK SONUÇLARI '))
               self.vhist()  
        if self.comboBox.currentText()=='vgg19':
            self.modeladi="vgg19.h5"
            self.label_5.setText(str('VGG19 GRAFİK SONUÇLARI '))
            self.vgg19()
            

        data_generator =  ImageDataGenerator(
                rescale = 1./255,
                validation_split=float(validation_split)
                )  
        train_generator = data_generator.flow_from_directory(self.path,
                target_size = (224,224),
                class_mode='categorical',   
                batch_size =int(batch_size),
                shuffle=True,
                subset='training'
                )
        
        test_generator = data_generator.flow_from_directory(
                self.path,
                class_mode='categorical',
                target_size = (224, 224),
                batch_size = int(batch_size),
                shuffle=False,
                subset='validation'
                )
        
        if self.comboBox_3.currentText()=="Adam":
            opt=tensorflow.keras.optimizers.Adam(lr=float(learning_rate))
        if self.comboBox_3.currentText()=="SGD":
            opt=tensorflow.keras.optimizers.SDG(lr=float(learning_rate))
        if self.comboBox_3.currentText()=="RMSprop":
            opt=tensorflow.keras.optimizers.Rmsprop(lr=float(learning_rate))
            
        self.model.compile(optimizer = opt,
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
            
        history=self.model.fit(
                train_generator,
                validation_data = test_generator,
                epochs = int(nb_epochs),
                steps_per_epoch = len(train_generator),
                validation_steps = len(test_generator)
                )
      
        print("Model Eğitildi.")
        scores = self.model.evaluate(test_generator, verbose=0)
        self.lineEdit.setText(("Model Başarısı: %.2f%%" % (scores[1]*100)))
        
        self.model.save(self.modeladi)
        
    
        plt.figure(figsize=(10,2.5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('./accuarcy.png')
        self.pixmap = QPixmap("./accuarcy.png") 
        self.label_6.setPixmap(self.pixmap)
        
        
        plt.figure(figsize=(10,2.5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')    
        plt.savefig('./loss.png')
        self.pixmap = QPixmap("./loss.png") 
        self.label_7.setPixmap(self.pixmap)

        
        
        
        Y_pred = self.model.predict_generator(test_generator)
        y_pred = np.argmax(Y_pred, axis=1)
        
        print(test_generator.classes)
        cm = confusion_matrix(test_generator.classes, y_pred)
        print(cm)
        plt.figure(figsize=(4.5,2.5))
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax,cmap=plt.cm.Blues)
        ax.set_xlabel('Tahmin');ax.set_ylabel('Gerçek'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Kedi', 'Köpek'])
        ax.yaxis.set_ticklabels(['Kedi', 'Köpek'],rotation=45)
        plt.savefig("./matrixmakine.png")
        self.pixmap = QPixmap("./matrixmakine.png")
        self.label_9.setPixmap(self.pixmap)
        
        
        
        fpr = {}
        tpr = {}
        thresholds ={}     
        n_class = 2
        Y_pred = self.model.predict_generator(test_generator)
       
        print(test_generator.classes)
        
        print(Y_pred)
        for i in range(n_class):            
            fpr[i], tpr[i], thresholds[i] = roc_curve(test_generator.classes, Y_pred[:,i], pos_label=i)    
        plt.figure(figsize=(4.5,2.5))
        plt.plot(fpr[0], tpr[0], color='blue', label='Kedi')
        plt.plot(fpr[1], tpr[1], color='green', label='Köpek')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig('./roccurve.png')
        plt.show()    
        self.pixmap = QPixmap("./roccurve.png") 
        self.label_8.setPixmap(self.pixmap)  

        
    def vhist(self):
        validation_split = self.comboBox_4.currentText()
        batch_size = self.comboBox_5.currentText()
        nb_epochs = self.comboBox_6.currentText()
        learning_rate = self.comboBox_7.currentText()
        
        model1 = VGG16(input_shape=[224,224]+ [3],weights = 'imagenet', include_top = False)
        model2 = applications.InceptionV3(input_shape=[224,224] + [3], weights='imagenet', include_top=False)
        for layer in model1.layers:
            layer.trainable = False
        for layer in model2.layers:
            layer.trainable = False  
    
        model1_out=model1.output
        model1_final=GlobalAveragePooling2D()(model1_out)
        model2_out=model2.output
        model2_final=GlobalAveragePooling2D()(model2_out)
        merged_model=Concatenate()([model1_final, model2_final])
        final_model=Dense(512,activation='relu')(merged_model)
        final_model=Dense(512,activation='relu')(final_model)
        final_model=Dense(2,activation='softmax')(final_model)
        model = Model(inputs=[model1.input, model2.input], outputs=final_model)
        
        if self.comboBox_3.currentText()=="Adam":
            opt=tensorflow.keras.optimizers.Adam(lr=float(learning_rate))
        if self.comboBox_3.currentText()=="SGD":
            opt=tensorflow.keras.optimizers.SDG(lr=float(learning_rate))
        if self.comboBox_3.currentText()=="RMSprop":
            opt=tensorflow.keras.optimizers.Rmsprop(lr=float(learning_rate))
        
        model.compile(loss = "binary_crossentropy", optimizer = opt,
                       metrics=['accuracy'])
        
        
        model.summary()
   
        img_height=224
        img_width=224
        
      
        data_datagen = ImageDataGenerator(
            rescale = 1./255,  
            validation_split=float(validation_split),
         )
        
      
        
        
        def generate_generator_multiple(generator,dir1,dir2,batch_size,img_height,img_width,subset):
          
            train_gen=data_datagen.flow_from_directory(
                             dir1,
                             class_mode='categorical',
                             target_size=(img_height,img_width),
                             batch_size=int(batch_size),
                             subset=subset,
                            #seed=7
                             )                    
            test_gen = data_datagen.flow_from_directory(
                          dir2,
                          class_mode='categorical',
                          target_size = (img_height,img_width),
                          batch_size = int(batch_size), 
                          subset=subset,
                           #seed=7               
                           )

            while True:
                X1i=train_gen.next()
                X2i=test_gen.next()
                yield [X1i[0],X2i[0]],X2i[1]

            
        inputgenerator=generate_generator_multiple(
                generator=data_datagen,
                dir1=self.path,
                dir2=self.path,       
                batch_size=int(batch_size),
                img_height=img_height,
                img_width=img_width,
                subset='training'          
                  
                )
        testgenerator=generate_generator_multiple(
            generator=data_datagen,
            dir1=self.path,
            dir2=self.path,
            batch_size=int(batch_size),
            img_height=img_height,
            img_width=img_width,
            subset='validation'          
            )
             
        history=model.fit(
                 inputgenerator,
                 steps_per_epoch=401,
                 epochs=int(nb_epochs),
                 validation_data=testgenerator,
                 shuffle=False,
                  validation_steps=101

                 )
        print("kaydettim")
        
        model.save(self.modeladi)
       
        score = model.evaluate_generator(testgenerator,101)
        self.lineEdit.setText(("Model Başarısı: %.2f%%" % (score[1]*100)))
        
     
       
        plt.figure(figsize=(10,2.5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('./accvhist.png')
        self.pixmap = QPixmap("./accvhist.png") 
        self.label_6.setPixmap(self.pixmap)
        
        
        plt.figure(figsize=(10,2.5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('./lossvhist.png')
        self.pixmap = QPixmap("./lossvhist.png") 
        self.label_7.setPixmap(self.pixmap)



        
        Y_pred = model.predict(testgenerator,steps=1)
        y_pred = np.argmax(Y_pred, axis=1)
  
        cm = confusion_matrix(testgenerator.classes, y_pred)

        print(cm)
        plt.figure(figsize=(4.5,2.5))
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax,cmap=plt.cm.Blues)
        ax.set_xlabel('Tahmin');ax.set_ylabel('Gerçek'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Kedi', 'Köpek'])
        ax.yaxis.set_ticklabels(['Kedi', 'Köpek'],rotation=45)
        plt.savefig("./matrixmakine.png")
        self.pixmap = QPixmap("./matrixmakine.png")
        self.label_9.setPixmap(self.pixmap)
        
        
        fpr = {}
        tpr = {}
        thresholds ={}     
        n_class = 2
        Y_pred = self.model.predict_generator(testgenerator,3)
    
        for i in range(n_class):            
            fpr[i], tpr[i], thresholds[i] = roc_curve(testgenerator.classes, Y_pred[:,i], pos_label=i)    
        plt.figure(figsize=(4.5,2.5))
        plt.plot(fpr[0], tpr[0], color='blue', label='Kedi')
        plt.plot(fpr[1], tpr[1], color='green', label='Köpek')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig('./roccurve.png')
        plt.show()    
        self.pixmap = QPixmap("./roccurve.png") 
        self.label_8.setPixmap(self.pixmap)  
        
        
    def aa(self):
      if self.comboBox_2.currentText()=="vgg16":
           kullanilacakmodel="vgg16.h5"
      if self.comboBox_2.currentText()=="Inception-V3":
           kullanilacakmodel="Inception-V3.h5"
           
      if self.comboBox_2.currentText()=="Vhist":
           kullanilacakmodel="Vhist.h5"
      if self.comboBox_2.currentText()=="vgg19":
           kullanilacakmodel="vgg19.h5"
           
      model=tf.keras.models.load_model(kullanilacakmodel)
      img = image.load_img(self.son,target_size=(224,224))
      img = np.asarray(img)
      plt.imshow(img)
      img = np.expand_dims(img, axis=0)
      output = model.predict(img)   
      print(output)
      enbuyuk=output[0][0]
      k=0
      for i in range(2):    
        if enbuyuk<output[0][i]:
            enbuyuk=output[0][i]
            k=i
      kedikopek=["Kedi","Köpek"]        
    
      self.lineEdit_4.setText("%.2f%%" % (enbuyuk*100)+" "+kedikopek[k])  
def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = window()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        
        
        
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        