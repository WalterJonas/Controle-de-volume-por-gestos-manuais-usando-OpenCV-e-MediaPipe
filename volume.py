import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inicialize os módulos de MediaPipe para mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure o objeto de detecção de mãos
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Configure o controle de volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Capture o vídeo da webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Não foi possível capturar a imagem da câmera")
        break

    # Converta a imagem para RGB (MediaPipe usa RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Inverter a imagem horizontalmente para um espelho de visualização
    image = cv2.flip(image, 1)

    # Para melhorar a performance, desative a gravação da imagem
    image.flags.writeable = False
    
    # Processar a imagem e encontrar mãos
    results = hands.process(image)
    
    # Gravação da imagem ativada
    image.flags.writeable = True

    # Converta a imagem de volta para BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Desenhe as anotações da mão na imagem
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordenadas do polegar e do dedo indicador
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Converter coordenadas normalizadas para coordenadas de pixel
            thumb_tip = np.array([thumb_tip.x * image.shape[1], thumb_tip.y * image.shape[0]])
            index_tip = np.array([index_tip.x * image.shape[1], index_tip.y * image.shape[0]])
            
            # Calcular a distância entre o polegar e o dedo indicador
            distance = np.linalg.norm(thumb_tip - index_tip)
            
            # Mapeie a distância para o controle de volume
            volume_range = volume.GetVolumeRange()  # (-65.25, 0.0, 0.03125)
            min_volume = volume_range[0]
            max_volume = volume_range[1]
            vol = np.interp(distance, [20, 200], [min_volume, max_volume])
            volume.SetMasterVolumeLevel(vol, None)
            
            # Desenhar uma linha entre o polegar e o dedo indicador
            cv2.line(image, tuple(thumb_tip.astype(int)), tuple(index_tip.astype(int)), (255, 0, 0), 3)
            cv2.putText(image, f'Volume: {int(np.interp(vol, [min_volume, max_volume], [0, 100]))} %', 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Mostrar a imagem
    cv2.imshow('Hand Volume Control', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
