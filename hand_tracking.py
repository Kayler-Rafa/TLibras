import cv2
import mediapipe as mp

# =========================
# SETUP
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Espelha
    frame = cv2.flip(frame, 1)

    # Converte para RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa
    results = hands.process(rgb)

    # =========================
    # DESENHO + TRACKING
    # =========================
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Desenha os 21 pontos + conexões
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # TRACKING REAL (exemplo: ponta do indicador)
            h, w, _ = frame.shape

            index_tip = hand_landmarks.landmark[8]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            cv2.putText(frame,
                        f"X:{x} Y:{y}",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2)

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow("Hand Tracking REAL", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()