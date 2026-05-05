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
import joblib
from sklearn.ensemble import RandomForestClassifier

ARQUIVO_DATASET = "gesto_gatilho_dataset.csv"
ARQUIVO_MODELO  = "gesto_gatilho.pkl"
MIN_AMOSTRAS    = 40   # por classe


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pts_para_vetor(landmarks, h, w):
    v = []
    for lm in landmarks:
        v.extend([int(lm.x * w), int(lm.y * h)])
    return v


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

def capturar_fase(cap, hands, rotulo, cor, instrucao):
    amostras = []
    print(f"\n{'='*50}")
    print(f"  {instrucao}")
    print(f"  Meta: {MIN_AMOSTRAS} amostras  |  ESPAÇO = capturar  |  ESC = pular fase")
    print(f"{'='*50}")

    while len(amostras) < MIN_AMOSTRAS:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        vetor = None
        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
            vetor = _pts_para_vetor(hl.landmark, h, w)

        # Overlay de instrução
        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.putText(frame, instrucao, (16, 38),
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Webcam não encontrada.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    print("\n=== Criação do Gesto Gatilho ===")
    print("Escolha um gesto que NÃO seja nenhuma letra/palavra do seu vocabulário.")
    print(f"Você precisará de pelo menos {MIN_AMOSTRAS} amostras por fase.\n")

    pos = capturar_fase(cap, hands,
                        "Gatilho", (0, 200, 0),
                        "FASE 1: Faca o GESTO GATILHO")

    neg = capturar_fase(cap, hands,
                        "Outros", (0, 120, 255),
                        "FASE 2: Faca OUTROS gestos (nao-gatilho)")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    if len(pos) < 5 or len(neg) < 5:
        print("\n[AVISO] Amostras insuficientes — modelo não foi gerado.")
        return

    treinar(pos, neg)
    print("\nPronto! Execute hand_tracking.py para usar o gesto gatilho.")


if __name__ == "__main__":
    main()
