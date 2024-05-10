import mediapipe as mp
import numpy
import cv2
import dlib
import time
import enum
# from cvzone.HandTrackingModule import FindPosition
# from cvzone.HandTrackingModule import HandDetector
# import HandTrackingModule as htm
from mediapipe.tasks import python

# from mediapipe.tasks.python import vision


cap = cv2.VideoCapture(0)
cascadeClassifierPath = 'haarcascade_frontalface_alt.xml' # Chemin du Classifier
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
hands = mp.solutions.hands #la solution est la main
hands_mesh = hands.Hands(static_image_mode=True, min_detection_confidence=0.7) #c'est une class qui attend des parametre
draw = mp.solutions.drawing_utils
pTime = 0

# mp_hands = mp.solutions.hands
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#         image = cv2.flip(cv2.imread(file), 1)
        
#         results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# detector = vision.HandLandmarker.create_from_options(options)


# detector = htm.handDetector(detectionCon=0.7)
# hand_landmarker = landmarker_and_result()
mp_hands = mp.solutions.hands.Hands()
while True and cap.isOpened():
    _,frame = cap.read()
    # img = frame
    # img = detector.findHands(img)
    # lmList = detector.findPosition(img, draw=False)
    # detection_result: mp.tasks.vision.HandLandmarkerResult
    # finger = mp.tasks.vision.HandLandmarkerResult.hand_landmarks
    # test = HandLandmark
    
    # results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # options = vision.HandLandmarkerOptions(
    #                                    num_hands=2)
    # detect = vision.HandLandmarker.create_from_options(options)
    # detectFinger = mp.tasks.vision.HandLandmarker.hand_landmarks[2]
    # detectFinger = mp.tasks.vision.hand_landmarks[2]
    # detectFinger = mp.tasks.vision.HandLandmarkerResult.hand_landmarks[2]
    # detectFinger = mp.tasks.vision.HandLandmarkerResult[2]
    # detectFinger = mp.tasks.vision.hand_landmark.landmark[0]

    # for hand_landmark in handFrame.multi_hand_landmarks: 
    results = mp_hands.process(frame)
    if results.multi_hand_landmarks:
        # Hands were detected
        # print('testestest')
        
        ...
    else:
        # No hands were detected
        ...


    # detectFinger = mp.tasks.vision.HandLandmarker.hand_landmark.landmark[0]
    # detectFinger = vision.HandLandmarkerResult[2]
    # print(detector)

    # print(finger)

    # if len(lmList) !=0:
    #     print(lmList[2])
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # cv2.putText(img,f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_COMPLEX,
    #             3, (255, 0, 0), 3 )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #au lieu du noir et blanc, les pixels detecte le rgb
    varGrayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detectedFaces = cascadeClassifier.detectMultiScale(varGrayImg, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

    op = hands_mesh.process(rgb)
    for(x,y, width, height) in detectedFaces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0,255,0), 3) # Dessin d'un rectangle
        cv2.putText(frame,"Charly", (x+width, y+height), cv2.FONT_HERSHEY_SIMPLEX,0.8,(125,0,56),2)
    # cv2.imshow("result", frame)
 
    if op.multi_hand_landmarks:
        for hand_landmarks in op.multi_hand_landmarks:
            # test = mp.tasks.vision.HandLandmarker.hand_landmarks[0]
            fingerX = hand_landmarks.landmark[12].x * 650
            fingerY = hand_landmarks.landmark[12].y * 460
            index = hand_landmarks.landmark[8].x
            indexY = hand_landmarks.landmark[8].y
            pouceY = hand_landmarks.landmark[4].y
            majeurY = hand_landmarks.landmark[12].y
            annulaireY = hand_landmarks.landmark[16].y
            oriculaireY = hand_landmarks.landmark[20].y

            indexY5 = hand_landmarks.landmark[7].y
            annulaireY13 = hand_landmarks.landmark[15].y
            oriculaireY17 = hand_landmarks.landmark[19].y
            majeurY11 = hand_landmarks.landmark[11].y
            # print('x : '+str(fingerX))
            # print('y : '+str(fingerY))
            # if fingerX and fingerY:
            if  indexY > indexY5 and  annulaireY > annulaireY13 and oriculaireY > oriculaireY17 and majeurY < majeurY11:
                cv2.rectangle(frame, (int(fingerX-60), int(fingerY)), (int(fingerX)+40,int(fingerY)+100), (255, 0, 0), -10)
            # cv2.rectangle(frame, (fingerX, fingerY), (fingerX+10, fingerY+10), (0,255,0), 3) # Dessin d'un rectangle
            # print(finger)
            draw.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS, landmark_drawing_spec=draw.DrawingSpec(color = (255,0,255), circle_radius=5)) # vas detecter qlql chose
    cv2.imshow("window", frame)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        cap.release()# a savoir
        break


# source : https://www.youtube.com/watch?v=aTt5s8K8KIc pour les mains