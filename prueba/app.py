from flask import Flask, render_template, Response
import cv2
import numpy as np
import pyautogui
import mediapipe as mp

app = Flask(__name__)

# Inicializar la detección de manos de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables de volumen
current_volume = 50  # Volumen inicial en porcentaje
increment = 5  # Incremento de volumen
max_volume = 100
min_volume = 0

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

def generate_frames():
    global current_volume  # Acceder a la variable global
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espejar la imagen para una interacción más natural
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convertir la imagen a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Detectar la mano y ajustar el volumen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los puntos de referencia de la mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener los puntos de referencia de los dedos pulgar e índice
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calcular la distancia entre las puntas del pulgar y el índice
                thumb_pos = np.array([thumb_tip.x * w, thumb_tip.y * h])
                index_pos = np.array([index_tip.x * w, index_tip.y * h])
                distance = np.linalg.norm(thumb_pos - index_pos)

                # Calcular el porcentaje de volumen basado en la distancia
                volume_percentage = np.clip((distance / 200) * 100, min_volume, max_volume)

                # Ajustar el volumen solo si se detecta un cambio significativo
                if distance > 50:
                    if volume_percentage > current_volume + increment:
                        current_volume = min(current_volume + increment, max_volume)
                    elif volume_percentage < current_volume - increment:
                        current_volume = max(current_volume - increment, min_volume)

                # Controlar el volumen usando pyautogui
                if volume_percentage != current_volume:
                    if volume_percentage > current_volume:
                        pyautogui.press("volumeup")
                    else:
                        pyautogui.press("volumedown")

                # Dibujar línea verde cuando los dedos están "abiertos"
                if distance > 50:
                    cv2.line(frame, tuple(index_pos.astype(int)), tuple(thumb_pos.astype(int)), (0, 255, 0), 3)

        # Convertir el marco en un formato que Flask puede enviar
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

