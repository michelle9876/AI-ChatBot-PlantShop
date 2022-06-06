# -*- coding: utf-8 -*-

#######################################################
# Import Library
#######################################################
import aiml
import csv
from sklearn.metrics.pairwise import cosine_similarity
import pandas
import sys
import random
import string
import nltk 
import numpy as np
import webbrowser
import math

from numpy import dot
from numpy.linalg import norm

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer


#######################################################
#  Initialise NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.inference import ResolutionProver

read_expr = Expression.fromstring


#######################################################
#  Import library for Speech Recognition
#######################################################
import speech_recognition
import pyttsx3
recognizer = speech_recognition.Recognizer()



#######################################################
# Fuzzy Logic Libray
#######################################################
from simpful import *

FS = FuzzySystem()


#######################################################
#  Initialise Knowledgebase. 
#######################################################
import pandas
import sys


kb=[]

data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
# >>> ADD SOME CODES here for checking KB integrity (no contradiction), 
# otherwise show an error message and terminate
#a1 = read_expr(r'Flower(x) <-> Plant(x) & Stem(x)')
#kb=[a1]
# Using KB Integrity Check

print("Checking integrity")
# using resolutionProver to check integrity : Resolution can also be used to check the KB integrity. (to check if there is any contradiction in it)
# Q= NULL
# if resolution is successful, it means the NULL sentence is necessarily true: something must have benn wrong with KB!
answer = ResolutionProver().prove(None, kb, verbose=True)

if answer:
    print("KB Integrity Check is failed! Check your KB file")
    sys.exit()
else :
    print("KB Integrity Check is processed. ")
    
# corpus : csv lines    
def getAllKeyWords(corpus,wordsInYourInput):
    
    allWords = []
    
    for i in range(len(corpus)):
        wordsInRow = corpus[i].split()
        
    for i in range(len(wordsInRow)):
        allWords.append(wordsInRow[i])
        
    for i in range(len(wordsInYourInput)):
        allWords.append(wordsInYourInput[i])
        
        # # get unique words from the arrays.
        # [allWords.append(x) for x in wordsInRow if x not in allWords]
    
        # [allWords.append(x) for x in wordsInYourInput if x not in allWords]
    allWords = list(set(allWords))
    #print(allWords)
    return allWords



#######################################################
# TF Methods(Term Frequency) 
#######################################################

# calculate TF values of the words in one line and put it in the dictionary.

def tfFunction(wordsInYourInput):
    TFdictionary = {}
    
    uniqueWords = list(set(wordsInYourInput))

    for i in range(len(uniqueWords)):
        counter = 0
        for j in range(len(wordsInYourInput)):
            if uniqueWords[i] == wordsInYourInput[j]:
                counter =counter + 1
        TFdictionary[uniqueWords[i]] = counter/len(wordsInYourInput)
    #print("TF", TFdictionary, "\n")
    
    return TFdictionary
    

#######################################################
# IDF Methods(Inverse Data Frequency)
#######################################################

# Generate idf values for each words and put it in the dictionary.
   
def idfFunction(allWords, corpus, userInput):
    allQuestions = []
    for i in range(len(corpus)):
        allQuestions.append(corpus[i])
    allQuestions.append(userInput)  
    NumOfSample = len(allQuestions)
    
    IDFdictionary = {}
    for i in range(len(allWords)):
        counter = 0
        
        for j in range(len(allQuestions)):
            split_List = list(set(allQuestions[j].split()))
            for k in range(len(split_List)):
                if split_List[k] == allWords[i]:
                    counter = counter + 1
                    break
        IDFdictionary[allWords[i]] = math.log10(NumOfSample/counter)
    
   # print("IDF", IDFdictionary, "\n")

    return IDFdictionary

    
#######################################################
# TFIDF Methods
#######################################################


def tfidfFunction(TFdictionary, IDFdictionary, allQuestions ):
    tfidfDictionary = dict.fromkeys(allQuestions, 0)
    
    for key1, value1 in tfidfDictionary.items():
        for key2, value2 in TFdictionary.items():
            if key2 == key1 :
                tfidfDictionary[key1] = value2 * IDFdictionary[key1]
                
   # print("TFIDF", tfidfDictionary, "\n")
    
    return tfidfDictionary


#######################################################
# Consine Similarity Methods
#######################################################

def cosineSimilarityFunction(tfidfInput,tfidfQuestion):
    tfidfInput = list(tfidfInput.values())
    tfidfQuestion = list(tfidfQuestion.values())
    
   # print(tfidfInput)
   # print(tfidfQuestion)
    x = (norm(tfidfInput)*norm(tfidfQuestion))
    if(x != 0):
        result = dot(tfidfInput, tfidfQuestion)/ x
    else:
        result = 0
   #print(result)
        
    return result

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.




#######################################################
# AI Image Classification
#######################################################

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

import PIL

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import PIL.Image
# from tkinter import *
import tkinter as tk
# import tkinter.filedialog as fd
import os
from tkinter import filedialog

batch_size = 32
img_height = 180
img_width = 180

def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(180, 180, 3))
    img_tensor = image.img_to_array(img)     
    img_tensor = np.expand_dims(img_tensor, axis=0)    
    return img_tensor


#######################################################
# Object detection
#######################################################

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from tkinter import filedialog
# %matplotlib inline
import torch

DetectionModel = torch.hub.load('yolov5', 'custom', path='FlowerModel.pt', source='local')  



#######################################################
# Welcome user
#######################################################

print("Welcome to Michelle's Plant shop chat bot. Please feel free to ask questions from me!")
print("")
kern.bootstrap(learnFiles="michelleFlower.xml")

#######################################################
# Main loop
#######################################################

userInput = ""
method = ""
userInput1 = "1"
userInput2= "2"

#######################################################
# Load model for Image classification
#######################################################
# model = load_model('C:/Users/kimbg/Desktop/AI_assignment/AI_assignment/model.h5')
#model = load_model('model.h5')

theLabel = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()
while True:
        print("")
        print("Please Select methods: Would you like to type the questions or use your voice?")
        print("[Type: 1 , Voice : 2] ")
        
        userchoice = input("> ")
        
        if (userInput1 == userchoice):
                method = "type"
                print("Type mode is selected")
                break
                
        elif (userInput2 == userchoice):
                method = "voice"
                print("Voice mode is selected")
                break
        else:
            print("Please type the number 1 or 2")
                
                
while True:
    
    
    if method == "type":
        #get user input
        try:
            print("")
            print("You may type your question. ")
            
            userInput = input("> ")
            
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break
        
 # code for speech recognition      
    elif method == "voice":
        try:
            with speech_recognition.Microphone() as mic:
                print("")
                print("Press s to start recording")
                
                while True:
                    
                    userchoice = input("> ")
                    
                    if (userchoice == "s"):
                        print("You may start to speak now")
                        print("")
                        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                        audio = recognizer.listen(mic)
                        
                        userInput = recognizer.recognize_google(audio)
                        userInput = userInput.lower()
                        
                        print(f"Recognized {userInput}")
                        break

        except speech_recognition.UnknownValueError():
            
            recognizer = speech_recognition.Recognizer()
            continue
            
        
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
                
        # Here are the processing of the new logical component:
            
        # if input pattern is "I know that * is *"
        elif cmd == 31: 
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            # >>> ADD SOME CODES HERE to make sure expr does not contradict 
            # with the KB before appending, otherwise show an error message.
            print("Chekcing contradict with KB ")
            kbCopy = kb.copy()
            kbCopy.append(expr)
            answer = ResolutionProver().prove(None, kbCopy, verbose=True)
            if answer:
                print("Sorry, it contradicts with KB")
            else :
                found = False
                for line in kb :
                    if(line == expr):
                        print("It is already exists in KB ")
                        found = True
                        break
                if(found == False):
                    print("Okay remembered. There is no contradicton in " + object + " is " + subject)
                    kb.append(expr) 
            
        # if the input pattern is "check that * is *"    
        elif cmd == 32: 
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('Checking that if it is false.')
               expr=read_expr('-' + subject + '(' + object + ')')
               answer = ResolutionProver().prove(expr, kb, verbose=True)
               
               if answer:
                   print ("It is incorrect")
               else :
                   print("Sorry I don't know.")
              
        # if input pattern is "I know that * is not *"       
        elif cmd == 33: 
            object,subject=params[1].split(' is not')
            expr=read_expr('-' + subject + '(' + object + ')')
            # >>> ADD SOME CODES HERE to make sure expr does not contradict 
            # with the KB before appending, otherwise show an error message.
            print("Chekcing contradict with KB ")
            kbCopy = kb.copy()
            kbCopy.append(expr)
            answer = ResolutionProver().prove(None, kbCopy, verbose=True)
            if answer:
                print("Sorry, it contradicts with KB")
            else :
                found = False
                for line in kb :
                    if(line == expr):
                        print("It is already exists in KB ")
                        found = True
                        break
                if(found == False):
                    print("Okay remembered. There is no contradicton in " + object + " is not" + subject)
                    kb.append(expr) 
                    
        # if the input pattern is "check that * is not *            
        elif cmd == 34: 
            object,subject=params[1].split(' is not ')
            expr=read_expr('-' + subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('Checking that if it is false.')
               # expr=read_expr('-' + subject + '(' + object + ')')
               expr=read_expr(subject + '(' + object + ')')
               answer = ResolutionProver().prove(expr, kb, verbose=True)
               
               if answer:
                   print ("It is incorrect")
               else :
                   print("Sorry I don't know.")
                   
        # if input pattern is "I know that * has *"            
        elif cmd == 35: 
            object,subject=params[1].split(' has ')
            expr=read_expr(subject + '(' + object + ')')
            
            print("Chekcing contradict with KB ")
            kbCopy = kb.copy()
            kbCopy.append(expr)
            answer = ResolutionProver().prove(None, kbCopy, verbose=True)
            if answer:
                print("Sorry, it contradicts with KB")
            else :
                found = False
                for line in kb :
                    if(line == expr):
                        print("It is already exists in KB ")
                        found = True
                        break
                if(found == False):
                    print("Okay remembered. There is no contradicton in " + object + " has " + subject)
                    kb.append(expr) 
                    
        # if the input pattern is "check that * has *            
        elif cmd == 36: 
            object,subject=params[1].split(' has ')
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('Checking that if it is false.')
               expr=read_expr('-' + subject + '(' + object + ')')
               answer = ResolutionProver().prove(expr, kb, verbose=True)
               
               if answer:
                   print ("It is incorrect")
               else :
                   print("Sorry I don't know.")
                   
        # if input pattern is "I know that * has not *"
        elif cmd == 37: 
            object,subject=params[1].split(' has not')
            expr=read_expr('-' + subject + '(' + object + ')')
            
            print("Chekcing contradict with KB ")
            kbCopy = kb.copy()
            kbCopy.append(expr)
            answer = ResolutionProver().prove(None, kbCopy, verbose=True)
            if answer:
                print("Sorry, it contradicts with KB")
            else :
                found = False
                for line in kb :
                    if(line == expr):
                        print("It is already exists in KB ")
                        found = True
                        break
                if(found == False):
                    print("Okay remembered. There is no contradicton in " + object + " has not" + subject)
                    kb.append(expr) 
                    
        # if the input pattern is "check that * has not *       
        elif cmd == 38: 
            object,subject=params[1].split(' has not ')
            expr=read_expr('-' + subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('Checking that if it is false.')
               # expr=read_expr('-' + subject + '(' + object + ')')
               expr=read_expr(subject + '(' + object + ')')
               answer = ResolutionProver().prove(expr, kb, verbose=True)
               
               if answer:
                   print ("It is incorrect")
               else :
                   print("Sorry I don't know.")
                   
        # Using fuzzy logic          
        elif cmd == 40: 
            while(True):
                print('Please give a Rate of your Plant_Leaf color: [0: very brown / 5: yellow / 10: very green]')
                x = input()
                if(0 <= int(x) and int(x) <= 10):
                    break
                else:
                    print("Please type number between 0~10")
            
            while(True):
                print("Please give a Rate of your Plants' Moisture Condition : [0: very dry / 10: very wet]")
                y = input()
                if(0<= int(y) and int(y) <= 10):
                    break
                else:
                    print("Please type number between 0~10")
            
                
            S_1 = FuzzySet(points=[[0., 1.],  [5., 0.]], term="brown")
            S_2 = FuzzySet(points=[[0., 0.], [5., 1.], [10., 0.]], term="yellow")
            S_3 = FuzzySet(points=[[5., 0.],  [10., 1.]], term="green")
            FS.add_linguistic_variable("Leaf_Color", LinguisticVariable([S_1, S_2, S_3], concept="Leaf_Color"))
            
            F_1 = FuzzySet(points=[[0., 1.],  [10., 0.]], term="dry")
            F_2 = FuzzySet(points=[[0., 0.],  [10., 1.]], term="wet") 
            FS.add_linguistic_variable("moisture", LinguisticVariable([F_1,F_2], concept="Moisture Level"))
            
            
            FS.set_crisp_output_value("dead", 1)
            FS.set_crisp_output_value("need_sunlight", 75 )
            FS.set_crisp_output_value("need_water", 45)
            FS.set_crisp_output_value("fresh", 99)
            
            
            R1 = "IF(Leaf_Color IS brown) AND (moisture IS dry) THEN (Fresh_Level IS dead)"
            R2 = "IF(Leaf_Color IS brown) AND (moisture IS wet) THEN (Fresh_Level IS need_sunlight)"
            R3 = "IF(Leaf_Color IS yellow) AND (moisture IS dry) THEN (Fresh_Level IS need_water)"
            R4 = "IF(Leaf_Color IS yellow) AND (moisture IS wet) THEN (Fresh_Level IS need_sunlight)"
            R5 = "IF(Leaf_Color IS green) AND (moisture IS dry) THEN (Fresh_Level IS need_water )"
            R6 = "IF(Leaf_Color IS green) AND (moisture IS wet) THEN (Fresh_Level IS fresh)"
            
            
            FS.add_rules([R1, R2, R3, R4, R5, R6])
            
            
            # Set antecedents values
            FS.set_variable("Leaf_Color", x)
            FS.set_variable("moisture", y)
            
            FS.plot_variable("Leaf_Color")
            FS.plot_variable("moisture")
            # Perform Sugeno inference and print output
            print(FS.Sugeno_inference(["Fresh_Level"]))   
        elif cmd == 50:
                        #get the path of the flower dataset
            dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
            data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                               fname='flower_photos',
                                               untar=True)
            data_dir = pathlib.Path(data_dir)
            ###################################
            
            

            
            #pass all the path of the specified flower into a list
            roses = list(data_dir.glob('roses/*'))
            sunflowers = list(data_dir.glob('sunflowers/*'))
            dandelion = list(data_dir.glob('dandelion/*'))
            daisy = list(data_dir.glob('daisy/*'))
            tulips = list(data_dir.glob('tulips/*'))
            
            
            
            #imagePath = 'C:/Users/kimbg/Desktop/AI_assignment/AI_assignment/test.jpg'
            #print(roses)
            #specify the flower that you want to check here
            #theFlower = str(roses[99])
            
            # root = tk.Tk()
            # root.geometry('320x240')
            # root.title('Tkinter Test')
             
            # frame = tk.Frame(root)
            # frame.pack()
            
            # root = tk.Tk()
            # root.withdraw()
             
            path1 = filedialog.askopenfilename(initialdir='/',title="select a file",
                                      filetypes =(("jpg Files","*.jpg"),
                                                  ("png Files","*.png")))
            
            # print(path1)
            # root.mainloop()
            
            theFlower = str(path1)
            # print("This is a flower Path !!!!" + theFlower)
            
            
            image = load_image(theFlower)
            theImage = plt.imread(theFlower)
            
            # img = tf.keras.utils.load_img(
            #     theFlower, target_size=(img_height, img_width)
            # )
            # img_array = tf.keras.utils.img_to_array(img)
            # img_array = tf.expand_dims(img_array, 0) # Create
            
            # plt.imshow(theImage)
            # plt.show()
            #plt.imshow(new_image)
            pred = model.predict(image)
            # print(pred)
            y_classes = pred.argmax(axis=-1)
            print("Model from local Machine >> It is a " + theLabel[y_classes[0]])
            
            
            project_id = 'e2033ccd-d20c-49fd-95e5-31b15d0a6328'
            cv_key = '157120bf51d84b708b547cd98cb917ba'
            cv_endpoint = 'https://t0116478-ai-image.cognitiveservices.azure.com/'
            
            model_name = 'flowers' # this must match the model name you set when publishing your model iteration (it's case-sensitive)!
            # print('Ready to predict using model {} in project {}'.format(model_name, project_id))
            
            
            #%matplotlib inline
            
            
            # Get the test images from the data/vision/test folder
            # test_folder = os.path.join('flower_data', 'flowers', 'test')
            # test_images = os.listdir(test_folder)
            
            # root = Tk()
            # root.title = ("Flowers")
            
            # def open():
            #     global my_image
            #     root.filename = filedialog.askopenfilename(initialdir = "/", title="Select a File", filetypes = (("jpg files", "*.jpg"), ("all files", "*.*")))
            #     my_label = Label(root, text = root.filename).pack()
            #     my_image = ImageTk.PhotoImage(Image.open(root.filename))
            #     my_image_label = Label(image=my_image).pack()
            #     print("Path::" + my_image)
            # my_btm = Button(root, text = "open File", command = open).pack()
            # root.mainloop()
            
            
            #######################################################
            # Open File Dialog and print the File Path
            #######################################################
            # root = tk.Tk()
            # root.geometry('320x240')
            # root.title('Tkinter Test')
             
            # frame = tk.Frame(root)
            # frame.pack()
             
            # path1 = fd.askopenfilename(initialdir='/',title="select a file",
            #                           filetypes =(("jpg Files","*.jpg"),
            #                                       ("png Files","*.png"),("all files","*.*")))
            
            # root.mainloop()
            
            # function to call when user press
            # the save button, a filedialog will
            # open and ask to save file
            
            
            image_path = path1
            #image_path = os.path.join('flower_data/flowers' , 'test', 'sunflower.jpg')
            # Create an instance of the prediction service
            credentials = ApiKeyCredentials(in_headers={"Prediction-key": cv_key})
            custom_vision_client = CustomVisionPredictionClient(endpoint=cv_endpoint, credentials=credentials)
            
            # Create a figure to display the results
            fig = plt.figure(figsize=(16, 8))
            
            # Get the images and show the predicted classes for each one
            #print('Classifying images in {} ...'.format(test_folder))
            # for i in range(len(test_images)):
            #     # Open the image, and use the custom vision model to classify it
            #     image_contents = open(os.path.join(test_folder, test_images[i]), "rb")
            #     image_stream = open(image_path, "rb")
                
            #     #classification = custom_vision_client.classify_image(project_id, model_name, image_contents.read())
            #     classification = custom_vision_client.classify_image(project_id, model_name, image_stream)
            #     # The results include a prediction for each tag, in descending order of probability - get the first one
            #     prediction = classification.predictions[0].tag_name
            #     # Display the image with its predicted class
            #     #img = Image.open(os.path.join(test_folder, test_images[i]))
            #     img = Image.open(image_path)
            #     #a=fig.add_subplot(len(test_images)/3, 3,i+1)
            #     a=fig.add_subplot(1,1,1)
            #     a.axis('off')
            #     imgplot = plt.imshow(img)
            #     a.set_title(prediction)
                # Open the image, and use the custom vision model to classify it
            image_stream = open(image_path, "rb")
            
            #classification = custom_vision_client.classify_image(project_id, model_name, image_contents.read())
            classification = custom_vision_client.classify_image(project_id, model_name, image_stream)
            # The results include a prediction for each tag, in descending order of probability - get the first one
            prediction = classification.predictions[0].tag_name
            # Display the image with its predicted class
            #img = Image.open(os.path.join(test_folder, test_images[i]))
            img = PIL.Image.open(image_path)
            #a=fig.add_subplot(len(test_images)/3, 3,i+1)
            a=fig.add_subplot(1,1,1)
            a.axis('off')
            imgplot = plt.imshow(img)
            a.set_title(prediction)
            plt.show()
            
            print('Model from Azure Cloud >> It is a' , prediction)
        elif cmd == 51:
            path1 = filedialog.askopenfilename(initialdir='/',title="select a file",
                          filetypes =(("jpg Files","*.jpg"),
                                      ("png Files","*.png")))

            results = DetectionModel(path1)
            
            # Results
            results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
            results.show()
            
            project_id = '0239b09f-7257-4f9f-8d3f-499dc3a2498d' # Replace with your project ID
            cv_key = '157120bf51d84b708b547cd98cb917ba' # Replace with your prediction resource primary key
            cv_endpoint = 'https://t0116478-ai-image.cognitiveservices.azure.com/' # Replace with your prediction resource endpoint
            
            model_name = 'FlowerObjectDetection' # this must match the model name you set when publishing your model iteration exactly (including case)!
            print('Ready to predict using model {} in project {}'.format(model_name, project_id))
            
            
            
            # # Load a test image and get its dimensions
            # path1 = filedialog.askopenfilename(initialdir='/',title="select a file",
            #                           filetypes =(("jpg Files","*.jpg"),
            #                                       ("png Files","*.png")))
            
            # test_img_file = os.path.join('data', 'object-detection', 'produce.jpg')
            
            test_img_file = path1
            
            # path1 = filedialog.askopenfilename(initialdir='/',title="select a file",
            #                           filetypes =(("jpg Files","*.jpg"),
            #                                       ("png Files","*.png")))
            
            test_img = Image.open(test_img_file)
            test_img_h, test_img_w, test_img_ch = np.array(test_img).shape
            
            # Get a prediction client for the object detection model
            credentials = ApiKeyCredentials(in_headers={"Prediction-key": cv_key})
            predictor = CustomVisionPredictionClient(endpoint=cv_endpoint, credentials=credentials)
            
            print('Detecting objects in {} using model {} in project {}...'.format(test_img_file, model_name, project_id))
            
            # Detect objects in the test image
            with open(test_img_file, mode="rb") as test_data:
                results = predictor.detect_image(project_id, model_name, test_data)
            
            # Create a figure to display the results
            fig = plt.figure(figsize=(8, 8))
            plt.axis('off')
            
            # Display the image with boxes around each detected object
            draw = ImageDraw.Draw(test_img)
            lineWidth = int(np.array(test_img).shape[1]/100)
            # object_colors = {
            #     "apple": "lightgreen",
            #     "banana": "yellow",
            #     "orange": "orange"
            # }
            
            fig = plt.figure(figsize=(16, 8))
            a=fig.add_subplot(1,1,1)
            a.axis('off')
            a.set_title("Prediction from Cloud")
            
            
            for prediction in results.predictions:
                color = 'white' # default for 'other' object tags
                if (prediction.probability*100) > 50:
                    # if prediction.tag_name in object_colors:
                    #     color = object_colors[prediction.tag_name]
                    left = prediction.bounding_box.left * test_img_w 
                    top = prediction.bounding_box.top * test_img_h 
                    height = prediction.bounding_box.height * test_img_h
                    width =  prediction.bounding_box.width * test_img_w
                    points = ((left,top), (left+width,top), (left+width,top+height), (left,top+height),(left,top))
                    draw.line(points, fill=color, width=lineWidth)
                    plt.annotate(prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100),(left,top), backgroundcolor=color)
            

            plt.imshow(test_img)
            plt.show()


            
            
        elif cmd == 99:
  # Open the csv file
            corpus = []
            anslist = []
            fileFlag = True
            try:
                file = open("plantShopQA1.csv")
            
            except IOError:
                input("Counld not open file! Please closed them")
                fileFlag = False
                
            if fileFlag is True:
                csvReader = csv.reader(file)
                for row in csvReader:
                    corpus.append(row[0])
                    anslist.append(row[1])
                    
                                
                #######################################################
                # TF-IDF Methods
                #######################################################
                
                #using TfidfTransformer you will first have to create a CountVectorizer to count 
                #the number of words (term frequency), limit your vocabulary size, apply stop words and etc.
                wordsInYourInput = userInput.split()
                allWords = getAllKeyWords(corpus, wordsInYourInput)
                #print(allWords)
                
                
                idf = idfFunction(allWords, corpus, userInput)
                
                tfInput = tfFunction(wordsInYourInput)
                
                
                tfidfInput = tfidfFunction(tfInput, idf, allWords)
                
                #print(tfidfInput)
                #print(allWords)
                                        
                #######################################################
                # Consine Similarity Methods
                #######################################################
                
            
                cosineSimilarityList = [None] * len(corpus)
                
                for i in range(len(corpus)):
                    
                    wordsIntheQuestion = corpus[i].split()
                    
                    tfOfQuestion = tfFunction(wordsIntheQuestion)
                    tfidfQuestion = tfidfFunction(tfOfQuestion, idf, allWords)
                    
                    cosineSimilarityList[i] = float(cosineSimilarityFunction(tfidfInput,tfidfQuestion))
                print("")
                print("Cosine Similarity of your line with the line on KB : ")
                print(cosineSimilarityList)
                maximumCosin = max(cosineSimilarityList)
                maximumIndex = cosineSimilarityList.index(maximumCosin)
                print("")
                print("")
                #If max value is 0, it means that no similar question is found with your line
                if maximumCosin == 0:
                    print("Sorry, we don't have a similar question in the KB ! ")
                else:
                    print("Your input question:", userInput, ", is similar to:")
                    print(corpus[maximumIndex])
                    print("")
                    print("Answer: ", anslist[maximumIndex])
                
                    file.close()
                
                
                
    else:
        print(answer)
        

