"""
Cria o gesto gatilho para o TLibras.

O gesto gatilho é um sinal específico que ativa o reconhecimento dinâmico:
ao detectá-lo, o programa captura os próximos 45 frames para identificar
uma letra ou palavra.

Uso:
    python criar_gesto_gatilho.py

Fase 1 — mostre o GESTO GATILHO e pressione ESPAÇO para capturar amostras.
Fase 2 — faça OUTROS gestos quaisquer e pressione ESPAÇO para amostras negativas.
Ao fim, o modelo é salvo em gesto_gatilho.pkl.
"""

import cv2
import mediapipe as mp
import csv
import os
import sys
import urllib.request
import joblib
from sklearn.ensemble import RandomForestClassifier
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ARQUIVO_DATASET = "gesto_gatilho_dataset.csv"
ARQUIVO_MODELO  = "gesto_gatilho.pkl"
MODEL_PATH      = "hand_landmarker.task"
MIN_AMOSTRAS    = 40   # por classe

# Conexões do esqueleto da mão (índices MediaPipe)
_CONEXOES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _garantir_modelo_mp():
    if not os.path.exists(MODEL_PATH):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print(f"Baixando {MODEL_PATH} ...")
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("Modelo baixado!")
        except Exception as e:
            print(f"[ERRO] Falha ao baixar: {e}")
            sys.exit(1)


def _criar_detector():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def _landmarks_para_pts(landmarks, h, w):
    return {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}


def _pts_para_vetor(pts):
    v = []
    for i in range(21):
        x, y = pts[i]
        v.extend([x, y])
    return v


def _desenhar_mao(frame, pts):
    for a, b in _CONEXOES:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
    for idx, (x, y) in pts.items():
        r = 6 if idx in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (80, 80, 80), 1, cv2.LINE_AA)


def _barra(frame, atual, total, rotulo, cor):
    h, w = frame.shape[:2]
    bw, bh = 320, 28
    x = (w - bw) // 2
    y = h - 70
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + int(bw * min(atual, total) / total), y + bh), cor, -1)
    cv2.putText(frame, f"{rotulo}: {atual}/{total}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# ── Coleta de fase ────────────────────────────────────────────────────────────

def capturar_fase(cap, detector, rotulo, cor, instrucao):
    amostras = []
    print(f"\n{'='*52}")
    print(f"  {instrucao}")
    print(f"  Meta: {MIN_AMOSTRAS} amostras  |  ESPAÇO=capturar  |  ESC=pular")
    print(f"{'='*52}")

    while len(amostras) < MIN_AMOSTRAS:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_img)

        vetor = None
        if result.hand_landmarks:
            pts = _landmarks_para_pts(result.hand_landmarks[0], h, w)
            _desenhar_mao(frame, pts)
            vetor = _pts_para_vetor(pts)

        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.putText(frame, instrucao, (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
        cv2.putText(frame, "ESPACO = capturar   ESC = pular fase", (16, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        _barra(frame, len(amostras), MIN_AMOSTRAS, rotulo, cor)

        cv2.imshow("Criar Gesto Gatilho", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("  Fase pulada.")
            break
        if key == ord(" ") and vetor:
            amostras.append(vetor)
            print(f"  Capturada {len(amostras)}/{MIN_AMOSTRAS}")

    return amostras


# ── Treino ────────────────────────────────────────────────────────────────────

def treinar(amostras_gatilho, amostras_outro):
    X = amostras_gatilho + amostras_outro
    y = ["gatilho"] * len(amostras_gatilho) + ["outro"] * len(amostras_outro)

    with open(ARQUIVO_DATASET, "w", newline="") as f:
        writer = csv.writer(f)
        for vetor, rotulo in zip(X, y):
            writer.writerow(vetor + [rotulo])
    print(f"\nDataset salvo em '{ARQUIVO_DATASET}' ({len(X)} amostras).")

    modelo = RandomForestClassifier(n_estimators=150, random_state=42)
    modelo.fit(X, y)
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"Modelo salvo em '{ARQUIVO_MODELO}'.")
    print(f"Acurácia no treino: {modelo.score(X, y):.0%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _garantir_modelo_mp()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Webcam não encontrada.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=== Criação do Gesto Gatilho ===")
    print("Escolha um gesto que NÃO seja nenhuma letra/palavra do seu vocabulário.")
    print(f"Você precisará de pelo menos {MIN_AMOSTRAS} amostras por fase.\n")

    with _criar_detector() as detector:
        pos = capturar_fase(cap, detector,
                            "Gatilho", (0, 200, 0),
                            "FASE 1: Faca o GESTO GATILHO")

        neg = capturar_fase(cap, detector,
                            "Outros", (0, 120, 255),
                            "FASE 2: Faca OUTROS gestos (nao-gatilho)")

    cap.release()
    cv2.destroyAllWindows()

    if len(pos) < 5 or len(neg) < 5:
        print("\n[AVISO] Amostras insuficientes — modelo não foi gerado.")
        return

    treinar(pos, neg)
    print("\nPronto! Execute hand_tracking.py para usar o gesto gatilho.")


if __name__ == "__main__":
    main()
