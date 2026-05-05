"""
Cria amostras de palavras/letras para reconhecimento de gestos dinâmicos.

Cada amostra captura 45 frames consecutivos de movimento.
O modelo treinado (modelo_palavras.pkl) é usado pelo hand_tracking.py para
reconhecer qual palavra foi sinalizada após o gesto gatilho.

Uso:
    python criar_palavra.py

Fluxo por amostra:
  1. Digite a palavra/letra no terminal.
  2. Posicione a mão e pressione ESPAÇO.
  3. Contagem regressiva de 3 segundos.
  4. Execute o gesto — 45 frames são gravados automaticamente.
  5. Repita para acumular mais amostras.
  6. Pressione ENTER sem texto para retreinar e sair.
"""

import cv2
import mediapipe as mp
import csv
import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier

ARQUIVO_DATASET  = "palavras_dataset.csv"
ARQUIVO_MODELO   = "modelo_palavras.pkl"
FRAMES_POR_GESTO = 45
CONTAGEM_SEG     = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pts_para_vetor(landmarks, h, w):
    v = []
    for lm in landmarks:
        v.extend([int(lm.x * w), int(lm.y * h)])
    return v


def _overlay_topo(frame, linha1, linha2="", cor1=(255, 255, 0)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 85), (0, 0, 0), -1)
    cv2.putText(frame, linha1, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, cor1, 2)
    if linha2:
        cv2.putText(frame, linha2, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def _barra_progresso(frame, atual, total):
    h, w = frame.shape[:2]
    progresso = int(w * min(atual, total) / total)
    cv2.rectangle(frame, (0, h - 44), (w, h), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, h - 44), (progresso, h), (0, 200, 0), -1)
    cv2.putText(frame, f"GRAVANDO: {atual}/{total} frames", (16, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ── Gravação de uma sequência ─────────────────────────────────────────────────

def gravar_sequencia(cap, hands, rotulo):
    """
    Exibe contagem regressiva e grava exatamente FRAMES_POR_GESTO frames.
    Retorna vetor achatado (1890 valores) ou None se a mão sumir.
    """
    # Contagem regressiva
    inicio = time.time()
    while time.time() - inicio < CONTAGEM_SEG:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS)

        restante = CONTAGEM_SEG - (time.time() - inicio)
        _overlay_topo(frame,
                      f"Palavra: {rotulo}",
                      f"Iniciando em {restante:.0f}...",
                      cor1=(0, 200, 255))
        cv2.imshow("Criar Palavra", frame)
        cv2.waitKey(1)

    # Gravação dos frames
    frames_coletados = []
    falhas = 0

    while len(frames_coletados) < FRAMES_POR_GESTO:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
            frames_coletados.append(_pts_para_vetor(hl.landmark, h, w))
            falhas = 0
        else:
            falhas += 1
            if falhas > 15:   # mais de 15 frames seguidos sem mão → cancela
                cv2.putText(frame, "MAO PERDIDA — cancelando", (60, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.imshow("Criar Palavra", frame)
                cv2.waitKey(800)
                return None

        _overlay_topo(frame, f"Palavra: {rotulo}", cor1=(255, 255, 0))
        _barra_progresso(frame, len(frames_coletados), FRAMES_POR_GESTO)
        cv2.imshow("Criar Palavra", frame)
        cv2.waitKey(1)

    # Achata: 45 frames × 42 valores = 1890
    entrada = []
    for vetor_frame in frames_coletados:
        entrada.extend(vetor_frame)
    return entrada


# ── Treino ────────────────────────────────────────────────────────────────────

def treinar_modelo():
    if not os.path.exists(ARQUIVO_DATASET):
        print("[AVISO] Nenhum dataset encontrado.")
        return

    linhas = []
    with open(ARQUIVO_DATASET, newline="") as f:
        for row in csv.reader(f):
            if row:
                linhas.append(row)

    if len(linhas) < 2:
        print("[AVISO] Amostras insuficientes para treinar (mínimo: 2).")
        return

    X = [[float(v) for v in row[:-1]] for row in linhas]
    y = [row[-1] for row in linhas]
    classes = sorted(set(y))

    print(f"\nTreinando: {len(X)} amostras | classes: {classes}")
    modelo = RandomForestClassifier(n_estimators=150, random_state=42)
    modelo.fit(X, y)
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"Modelo salvo em '{ARQUIVO_MODELO}' | Acurácia: {modelo.score(X, y):.0%}")


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

    print("\n=== Criação de Palavras / Gestos Dinâmicos ===")
    print(f"Cada amostra = {FRAMES_POR_GESTO} frames (~{FRAMES_POR_GESTO/30:.1f}s de gesto).")
    print("Recomendado: 20+ amostras por palavra para boa precisão.")
    print("ENTER sem texto → treina modelo e sai.\n")

    total_novas = 0

    while True:
        rotulo = input("Palavra/letra (ENTER para treinar e sair): ").strip()
        if not rotulo:
            break

        print(f"\n  Capturando '{rotulo}'. Posicione a mão e pressione ESPAÇO.")
        print("  ESC = voltar para menu de palavras.\n")

        # Feed de espera
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.multi_hand_landmarks[0],
                    mp.solutions.hands.HAND_CONNECTIONS)

            _overlay_topo(frame, f"Palavra: {rotulo}",
                          "ESPACO = gravar 45 frames   ESC = outra palavra")
            cv2.imshow("Criar Palavra", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key == ord(" "):
                entrada = gravar_sequencia(cap, hands, rotulo)
                if entrada:
                    with open(ARQUIVO_DATASET, "a", newline="") as f:
                        csv.writer(f).writerow(entrada + [rotulo])
                    total_novas += 1
                    print(f"  Amostra #{total_novas} salva para '{rotulo}'.")
                else:
                    print("  Amostra descartada (mão perdida).")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    if total_novas > 0:
        print(f"\n{total_novas} novas amostras adicionadas.")
        treinar_modelo()
    else:
        print("\nNenhuma nova amostra — modelo não alterado.")

    print("Pronto!")


if __name__ == "__main__":
    main()
