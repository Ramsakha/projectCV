# projectCV
extract the folder
execute gui.py in spyder or pycharm directing to extracted folder

# Consists three python files 
1-for training images(faces)
2-for recognizing trained images(faces)
3-for creating an user interface (to connect training and recognition files)

# In starting file -this is training file:
images are taken with webcam
   
      vid = cv2.VideoCapture(0)
      ret,frame = vid.read()
      
using haarcascade classifiers faces are detected on the image and faces are taken separately and converted from BGR to GRAY scale

      face_casade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
       faces = face_casade.detectMultiScale(gray,1.3,5)
       
If the multiple faces are present in an Images use the nearest face to find the nearest face use max area of the face with w and h detected using haarcascade clasifier

      for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                areas.append(w*h)
      idx = areas.index(max(areas))
      face_sv = faces[idx]
      x = face_sv[0]
      y = face_sv[1]
      w = face_sv[2]
      h = face_sv[3]
      face = gray[y:y+h,x:x+w]
 
 Now resize the face taken to a constant size 112X92 
 
     face_res = cv2.resize(face,(112,92))
 this step is to give the input with constant dimension i.e all faces with a dimension of 112X92 and to get better results
 
 create a folder inside the working directory and store the image folder name should be of face name (person name) which can be obtained with GUI 
 take 100 of faces of same person and store in the folder created
 and name image with 0 - 99.jpg/.png respectively
 
 we can also create a separate file for images inside working directory and create sub-folders for storing 100 faces of a person
 
 create a EigenFace recognition model
        
       model = cv2.face.EigenFaceRecognizer_create()
       
Now go through all sub-folders inside images folder and make it label for sub-folders from 0 
Now create tags of images and labels by going through all the images and label with the folder containing images
 
        imgs = []
        tags = []
        index = 0
        for (subdirs, dirs, files) in os.walk(root):
            for subdir in dirs:
                img_path = os.path.join(root, subdir)
                for fn in os.listdir(img_path):
                    path = img_path + '/' + fn
                    tag = index
                    imgs.append(cv2.imread(path, 0))
                    tags.append(int(tag))
                index += 1
        (imgs, tags) = [np.array(item) for item in [imgs, tags]]
 
 Now train the model with input as imgs and tags our target is tag  and save the model with a name with an extension .xml(xml file)
 
          imags,tags = train_images(root)
          model.save('eigen_feature_data.xml')
 
 
 
 # End.py - recognition file
 first step is to load the model created in training file and stored
 
          face_casade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
          model = cv2.face.EigenFaceRecognizer_create()
          model.read('eigen_feature_data.xml')
 
 now detect the face in the frame with cascade classifier and convert to GRAY scale
           
          gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
          faces = face_casade.detectMultiScale(gray,1.3,5)

reshape the face to 112X92 
          
           gray_f = cv2.resize(gray_f,(112,92))
Now predict the face with the model created using .predict method
           
           confi = model.predict(gray_f)
This method gives confidence value we model is more confident with low value return by .predict method generally here used up to 3500
          
          if confi[1]<3500:
                    person = names[confi[0]]
                    cv2.putText(frame, '%s - %.0f' % (person, confi[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
          else:
                    person = 'Unknown'
                    cv2.putText(frame, person, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
 
 confi value is a tuple with 2 element in index 0 contains label predicted and in index 1 it contains confidence value
 putText is to display the detected name on the top of the(any where we wanted) image
 
 # GUI file(GUI.py)
 This is for interface for the user
 GUI created with tkinter
 user interface of size 900X610
          
           root.geometry('{}x{}'.format(900, 610))
           root.title('my face recognition model')
using favicon (just for showcase)
           
           root.iconbitmap('favicon.ico')
           
to take user input i.e image name 

           svalue = StringVar()
           w = Entry(root,textvariable=svalue) # adds a textarea widget
           w.pack()
           w.place(x=500,y=300)
    
    
to acess training and recognition files create two buttons with functionality and for train button make it to send name in text box entered by input to traing file with os module

           def train_fn():
                name = w.get()
                os.system('python starting.py %s'%name)

           def recog_fn():
               os.system('python end.py')

          train_button = Button(root,text="train", command=train_fn)
          train_button.pack()
          train_button.place(x=599,y=350)
          recog_button = Button(root,text="recognize", command=recog_fn)
          recog_button.pack()
          recog_button.place(x=565,y=400)
          
          
we can add images and favicon to the GUI created 
Using EigenFace recognizer results in faster training and better results with minium overfit
With using DEEPCNN takes times to train and recognize 
With 100 faces added every time training time is increased - so eigenfacerecognizer have advantage over DeepCNN

