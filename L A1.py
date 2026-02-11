import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
tangan = mp_hands.Hands(max_num_hands=2)
mpdraw = mp.solutions.drawing_utils

while True:
    success, frame = capture.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = tangan.process(imgRGB)

    if results.multi_hand_landmarks:
        for idx, titiktangan in enumerate(results.multi_hand_landmarks):
            mpdraw.draw_landmarks(
                frame, titiktangan, mp_hands.HAND_CONNECTIONS
            )

            #ambil titik pergelangan (landmarks 0) untuk posisi teks
            h, w, c = frame.shape
            x = int(titiktangan.landmark[0].x*w)
            y = int(titiktangan.landmark[0].y*h)

            # klasifikasi kiri / kanan
            hand = results.multi_handedness[idx]
            label = hand.classification[0].label # "left/"right"

            if label == "Right":
                cv2.putText(frame,"LEFT",(x, y - 20),
                            cv2.QT_FONT_NORMAL, 3, (255, 0, 0), 3)
            else:
                cv2.putText(frame, "RIGHT", (x, y - 20),
                            cv2.QT_FONT_NORMAL, 3, (0, 0, 255),3)

    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()