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
import sys
import time
import urllib.request
import joblib
from sklearn.ensemble import RandomForestClassifier
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ARQUIVO_DATASET  = "palavras_dataset.csv"
ARQUIVO_MODELO   = "modelo_palavras.pkl"
MODEL_PATH       = "hand_landmarker.task"
FRAMES_POR_GESTO = 45
CONTAGEM_SEG     = 3

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


def _detectar(detector, frame):
    """Retorna pts dict ou None."""
    h, w = frame.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)
    if result.hand_landmarks:
        return _landmarks_para_pts(result.hand_landmarks[0], h, w)
    return None


# ── Gravação de uma sequência ─────────────────────────────────────────────────

def gravar_sequencia(cap, detector, rotulo):
    """Contagem regressiva + 45 frames. Retorna vetor de 1890 valores ou None."""
    h_frame, w_frame = None, None

    # Contagem regressiva
    inicio = time.time()
    while time.time() - inicio < CONTAGEM_SEG:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]

        pts = _detectar(detector, frame)
        if pts:
            _desenhar_mao(frame, pts)

        restante = CONTAGEM_SEG - (time.time() - inicio)
        _overlay_topo(frame,
                      f"Palavra: {rotulo}",
                      f"Iniciando em {restante:.0f}...",
                      cor1=(0, 200, 255))
        cv2.imshow("Criar Palavra", frame)
        cv2.waitKey(1)

    # Gravação dos frames
    frames_coletados = []
    falhas_seguidas = 0

    while len(frames_coletados) < FRAMES_POR_GESTO:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        pts = _detectar(detector, frame)
        if pts:
            _desenhar_mao(frame, pts)
            frames_coletados.append(_pts_para_vetor(pts))
            falhas_seguidas = 0
        else:
            falhas_seguidas += 1
            if falhas_seguidas > 15:
                h, w = frame.shape[:2]
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
    _garantir_modelo_mp()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Webcam não encontrada.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=== Criação de Palavras / Gestos Dinâmicos ===")
    print(f"Cada amostra = {FRAMES_POR_GESTO} frames (~{FRAMES_POR_GESTO/30:.1f}s de gesto).")
    print("Recomendado: 20+ amostras por palavra para boa precisão.")
    print("ENTER sem texto → treina modelo e sai.\n")

    total_novas = 0

    with _criar_detector() as detector:
        while True:
            rotulo = input("Palavra/letra (ENTER para treinar e sair): ").strip()
            if not rotulo:
                break

            print(f"\n  Capturando '{rotulo}'. Posicione a mão e pressione ESPAÇO.")
            print("  ESC = voltar para menu de palavras.\n")

            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)

                pts = _detectar(detector, frame)
                if pts:
                    _desenhar_mao(frame, pts)

                _overlay_topo(frame, f"Palavra: {rotulo}",
                              "ESPACO = gravar 45 frames   ESC = outra palavra")
                cv2.imshow("Criar Palavra", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break

                if key == ord(" "):
                    entrada = gravar_sequencia(cap, detector, rotulo)
                    if entrada:
                        with open(ARQUIVO_DATASET, "a", newline="") as f:
                            csv.writer(f).writerow(entrada + [rotulo])
                        total_novas += 1
                        print(f"  Amostra #{total_novas} salva para '{rotulo}'.")
                    else:
                        print("  Amostra descartada (mão perdida).")

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
