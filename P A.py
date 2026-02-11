import cv2  
import mediapipe as mp 

# Inisialisasi webcam
capture = cv2.VideoCapture(0)  

# Inisialisasi deteksi tangan
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()  
while True:
   
    success, img = capture.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hand.process(imgRGB)
    
    if results.multi_hand_landmarks:
        print("tangan")  
    else:
        print("tidak ada") 

    cv2.imshow("webcam", img)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release() 
cv2.destroyAllWindows()  