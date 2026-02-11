import cv2  
import mediapipe

capture = cv2.VideoCapture(0)

mediapipehand = mediapipe.solutions.hands
tangan = mediapipehand.Hands(max_num_hands=1)
mpdraw = mediapipe.solutions.drawing_utils

while True:
    success, img = capture.read()  
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = tangan.process(imgRGB)

    if result.multi_hand_landmarks:
        for titiktangan in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, titiktangan, mediapipehand.HAND_CONNECTIONS)
            for id, titik in enumerate(titiktangan.landmark):
                print(id)
                print(titik.x)
                print(titik.y)

    cv2.imshow("webcam", img)
    
   
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()
