import cv2
import mediapipe as mp

# Inisialisasi kamera
camera = cv2.VideoCapture(0)

mp_hand = mp.solutions.hands
detector = mp_hand.Hands(max_num_hands=1)
util_gambar = mp.solutions.drawing_utils

while camera.isOpened():
    ok, img = camera.read()
    if not ok:
        continue

    #membalik agar seperti cermin
    img = cv2.flip(img, 1)

    #konversi warna RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hasil = detector.process(rgb_img)

    tinggi, lebar, _ = img.shape

    if hasil.multi_hand_landmarks:
        for tangan in hasil.multi_hand_landmarks:

            #Gambar kerangka tangan
            util_gambar.draw_landmarks(
                img, tangan, mp_hand.HAND_CONNECTIONS
            )

            #Ambil titik jempol (4) dan kelingking (20)
            titik_jempol = tangan.landmark[4]
            titik_kelingking = tangan.landmark[20]

            #Ubah ke koordinat pixel
            x_jempol = int(titik_jempol.x * lebar)
            x_kelingking = int(titik_kelingking.x * lebar)

            #Logika deteksi
            status_tangan = ""
            if x_jempol < x_kelingking:
                status_tangan = "Telapak_Tangan"
            else:
                status_tangan = "Punggung_Tangan"

            #Tampilkan Teks
            cv2.putText(
                img,
                status_tangan,
                (30,50),
                cv2.QT_FONT_NORMAL,
                1,
                (0,255,255),
                2
            )

    cv2.imshow("Deteksi Tangan", img)
    if cv2.waitKey(1) & 0xFF == 27 : #tekan esc untuk keluar
        break