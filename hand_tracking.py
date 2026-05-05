"""
Hand Tracking com MediaPipe + OpenCV
Compatível com mediapipe >= 0.9 e >= 0.10 (detecta automaticamente).

Instalação:
    pip install opencv-python mediapipe==0.10.9

Uso:
    python hand_tracking.py

Controles:
    Q  - Sair
    S  - Salvar screenshot
    D  - Alternar modo debug (exibe IDs das landmarks)
    +  - Aumentar espessura das linhas
    -  - Diminuir espessura das linhas
"""

import cv2
import mediapipe as mp
import time
import os
import sys
import joblib
import urllib.request as _urllib_req
import json

model = joblib.load("modelo.pkl")

# ── Modelos de gesto dinâmico ──────────────────
modelo_gatilho = None
modelo_palavras = None

if os.path.exists("gesto_gatilho.pkl"):
    modelo_gatilho = joblib.load("gesto_gatilho.pkl")
    print("Gesto gatilho carregado.")
else:
    print("[AVISO] gesto_gatilho.pkl não encontrado. Execute criar_gesto_gatilho.py.")

if os.path.exists("modelo_palavras.pkl"):
    modelo_palavras = joblib.load("modelo_palavras.pkl")
    print(f"Modelo de palavras carregado: {list(modelo_palavras.classes_)}")
else:
    print("[AVISO] modelo_palavras.pkl não encontrado. Execute criar_palavra.py.")

# ── Servidor de desenvolvimento ────────────────
SERVER_URL = "http://localhost:5000"
CONFIANCA_MINIMA = 0.80  # 80%

def enviar_palavra(letra, confianca, palavra):
    try:
        payload = json.dumps({
            "letra": letra,
            "confianca": confianca,
            "palavra": palavra,
        }).encode()
        req = _urllib_req.Request(
            SERVER_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        _urllib_req.urlopen(req, timeout=0.5)
    except Exception:
        pass  # servidor offline não trava o programa

#######################################################
import csv

def salvar_dados(pts, letra):
    linha = []

    for i in range(21):
        x, y = pts[i]
        linha.extend([x, y])

    linha.append(letra)

    with open("dataset.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(linha)
########################################################

# ── Detecta versão da API ─────────────────────
mp_version = tuple(int(x) for x in mp.__version__.split(".")[:2])
USE_NEW_API = mp_version >= (0, 10)
print(f"MediaPipe {mp.__version__} → usando {'nova API (>=0.10)' if USE_NEW_API else 'API clássica (<0.10)'}")

# ──────────────────────────────────────────────
#  Cores por dedo (BGR)
# ──────────────────────────────────────────────
FINGER_COLORS = {
    "thumb":  (255, 100,  50),
    "index":  ( 50, 255, 100),
    "middle": ( 50, 100, 255),
    "ring":   (255,  50, 200),
    "pinky":  (  0, 220, 220),
    "palm":   (200, 200, 200),
}

CONNECTIONS = {
    "thumb":  [(0,1),(1,2),(2,3),(3,4)],
    "index":  [(0,5),(5,6),(6,7),(7,8)],
    "middle": [(0,9),(9,10),(10,11),(11,12)],
    "ring":   [(0,13),(13,14),(14,15),(15,16)],
    "pinky":  [(0,17),(17,18),(18,19),(19,20)],
    "palm":   [(5,9),(9,13),(13,17)],
}


def landmarks_to_pts(landmarks, h, w):
    return {idx: (int(lm.x * w), int(lm.y * h)) for idx, lm in enumerate(landmarks)}


def draw_hand(frame, pts, thickness=2, debug=False):
    for finger, conns in CONNECTIONS.items():
        color = FINGER_COLORS[finger]
        for a, b in conns:
            cv2.line(frame, pts[a], pts[b], color, thickness, cv2.LINE_AA)
    for idx, (x, y) in pts.items():
        r = thickness + (5 if idx in (4, 8, 12, 16, 20) else 3)
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (80, 80, 80), 1, cv2.LINE_AA)
        if debug:
            cv2.putText(frame, str(idx), (x+6, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220,220,0), 1, cv2.LINE_AA)


def draw_label(frame, pts, label):
    wx, wy = pts[0]
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (wx-4, wy-lh-10), (wx+lw+4, wy-2), (30,30,30), -1)
    cv2.putText(frame, label, (wx, wy-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,180), 1, cv2.LINE_AA)


def draw_hud(frame, fps, num_hands, thickness, debug):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8,8), (265,122), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate([
        f"FPS: {fps:.1f}",
        f"Maos detectadas: {num_hands}",
        f"Espessura: {thickness}  (+/- para ajustar)",
        f"Debug: {'ON' if debug else 'OFF'}  (D para alternar)",
        "Q=sair  S=screenshot",
    ]):
        cv2.putText(frame, text, (14, 30+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════
#  API CLÁSSICA  (mediapipe < 0.10)
# ══════════════════════════════════════════════
def run_old_api(cap, state):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        num_hands = 0

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            for i, hl in enumerate(results.multi_hand_landmarks):
                pts = landmarks_to_pts(hl.landmark, h, w)
                draw_hand(frame, pts, state["thickness"], state["debug"])

                # DETECÇÃO DE GESTO
                texto, confianca = prever_letra(pts)

                # MOSTRAR TEXTO
                if texto:
                    x, y = pts[0]
                    cv2.putText(frame, f"{texto} {confianca:.0%}", (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # LABEL DA MÃO (Left/Right)
                if results.multi_handedness:
                    c = results.multi_handedness[i].classification[0]
                    draw_label(frame, pts, f"{c.label} ({c.score:.0%})")

        # FPS E HUD
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        draw_hud(frame, fps, num_hands, state["thickness"], state["debug"])
        cv2.imshow("Hand Tracking", frame)

        if handle_keys(cv2.waitKey(1) & 0xFF, state, frame):
            break

    hands.close()


# ══════════════════════════════════════════════
#  NOVA API  (mediapipe >= 0.10)
# ══════════════════════════════════════════════
def run_new_api(cap, state, palavra, ultima_letra, ultimo_tempo):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request

    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print(f"Baixando modelo MediaPipe Tasks em '{model_path}' ...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Modelo baixado!")
        except Exception as e:
            print(f"[ERRO] Falha ao baixar modelo: {e}")
            sys.exit(1)

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Máquina de estados ─────────────────────────────
    IDLE, GRAVANDO, RESULTADO = "idle", "gravando", "resultado"
    estado       = IDLE
    buffer_seq   = []      # pts dicts acumulados (máx. FRAMES_GESTO)
    res_rotulo   = ""
    res_conf     = 0.0
    res_tempo    = 0.0
    FRAMES_GESTO = 45
    TEMPO_RESULT = 2.5    # segundos exibindo o resultado

    prev_time = time.time()

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = detector.detect(mp_img)
            num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
            key = cv2.waitKey(1) & 0xFF

            # ── Lógica de estado ───────────────────────
            if result.hand_landmarks and estado != RESULTADO:
                pts = landmarks_to_pts(result.hand_landmarks[0], h, w)
                draw_hand(frame, pts, state["thickness"], state["debug"])

                if result.handedness:
                    cat = result.handedness[0][0]
                    draw_label(frame, pts, f"{cat.display_name} ({cat.score:.0%})")

                if estado == IDLE:
                    is_trig, conf_trig = prever_gatilho(pts)
                    _desenhar_indicador_gatilho(frame, conf_trig, w, h)
                    if is_trig:
                        estado = GRAVANDO
                        buffer_seq = []
                        print("Gatilho detectado — gravando sequência...")

                elif estado == GRAVANDO:
                    buffer_seq.append(dict(pts))
                    _desenhar_progresso_gravacao(frame, len(buffer_seq), FRAMES_GESTO, w, h)
                    if len(buffer_seq) >= FRAMES_GESTO:
                        res_rotulo, res_conf = classificar_sequencia(buffer_seq)
                        res_tempo = time.time()
                        estado = RESULTADO
                        print(f"Reconhecido: '{res_rotulo}' ({res_conf:.0%})")
                        if res_conf >= CONFIANCA_MINIMA:
                            palavra.append(res_rotulo)
                            enviar_palavra(res_rotulo, res_conf, "".join(palavra))

            elif estado == GRAVANDO and not result.hand_landmarks:
                cv2.putText(frame, "MAO NAO DETECTADA", (w // 2 - 170, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            # ── Exibição de resultado ──────────────────
            if estado == RESULTADO:
                _desenhar_resultado(frame, res_rotulo, res_conf, w, h)
                if time.time() - res_tempo > TEMPO_RESULT:
                    estado = IDLE

            # ── Palavra acumulada (rodapé) ─────────────
            texto_palavra = "".join(palavra)
            cv2.rectangle(frame, (30, h - 80), (w - 30, h - 20), (0, 0, 0), -1)
            cv2.putText(frame, texto_palavra, (40, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ── Estado no canto superior esquerdo ──────
            cor_estado = {IDLE: (100, 100, 100), GRAVANDO: (0, 200, 0), RESULTADO: (0, 200, 255)}
            cv2.putText(frame, estado.upper(), (14, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_estado[estado], 1)

            # ── FPS + HUD ──────────────────────────────
            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            draw_hud(frame, fps, num_hands, state["thickness"], state["debug"])
            cv2.imshow("Hand Tracking", frame)

            # ── Teclado ────────────────────────────────
            acao = handle_keys(key, state, frame)
            if acao is True:
                break
            elif acao == "reset":
                palavra.clear()
                estado = IDLE
                buffer_seq = []
                print("Palavra e estado reiniciados")
            elif acao == "delete":
                if palavra:
                    palavra.pop()
                    print("Apagou última letra")

# ══════════════════════════════════════════════
#  Teclado
# ══════════════════════════════════════════════
def handle_keys(key, state, frame):
    """Retorna True se deve encerrar."""
    if key == ord("q"):
        print("Encerrando...")
        return True

    elif key == ord("r"):  # 🔥 NOVO (RESET)
        return "reset"

    elif key == ord("x"):  # 🔥 NOVO (APAGAR ÚLTIMA)
        return "delete"

    elif key == ord("s"):
        os.makedirs("screenshots", exist_ok=True)
        fn = os.path.join("screenshots", f"hand_{int(time.time())}.png")
        cv2.imwrite(fn, frame)
        print(f"Screenshot salvo: {fn}")

    elif key == ord("d"):
        state["debug"] = not state["debug"]
        print(f"Debug: {'ON' if state['debug'] else 'OFF'}")

    elif key in (ord("+"), ord("=")):
        state["thickness"] = min(state["thickness"] + 1, 8)

    elif key == ord("-"):
        state["thickness"] = max(state["thickness"] - 1, 1)

    return False




def is_hand_open(pts):
    fingers = []

    # Indicador
    fingers.append(pts[8][1] < pts[6][1])
    # Médio
    fingers.append(pts[12][1] < pts[10][1])
    # Anelar
    fingers.append(pts[16][1] < pts[14][1])
    # Mindinho
    fingers.append(pts[20][1] < pts[18][1])

    return all(fingers)

def is_hand_closed(pts):
    return (
        pts[8][1] > pts[6][1] and
        pts[12][1] > pts[10][1] and
        pts[16][1] > pts[14][1] and
        pts[20][1] > pts[18][1]
    )


def is_index_up(pts):
    return (
        pts[8][1] < pts[6][1] and
        pts[12][1] > pts[10][1] and
        pts[16][1] > pts[14][1] and
        pts[20][1] > pts[18][1]
    )

def _pts_para_entrada(pts):
    entrada = []
    for i in range(21):
        x, y = pts[i]
        entrada.extend([x, y])
    return entrada


def prever_gatilho(pts):
    """Retorna (bool detectado, confiança). Usa gesto_gatilho.pkl."""
    if modelo_gatilho is None:
        return False, 0.0
    proba = modelo_gatilho.predict_proba([_pts_para_entrada(pts)])[0]
    classes = list(modelo_gatilho.classes_)
    if "gatilho" not in classes:
        return False, 0.0
    conf = float(proba[classes.index("gatilho")])
    return conf >= CONFIANCA_MINIMA, conf


def classificar_sequencia(buffer):
    """Classifica buffer de 45 pts-dicts. Retorna (rotulo, confiança)."""
    if modelo_palavras is None:
        return "?", 0.0
    entrada = []
    for pts in buffer:
        entrada.extend(_pts_para_entrada(pts))
    proba = modelo_palavras.predict_proba([entrada])[0]
    idx = proba.argmax()
    return str(modelo_palavras.classes_[idx]), float(proba[idx])


# ── Overlays de estado ─────────────────────────

def _desenhar_indicador_gatilho(frame, conf, w, h):
    """Mostra barra de confiança do gesto gatilho (canto inferior direito)."""
    bw, bh = 200, 18
    x, y = w - bw - 20, h - 60
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (40, 40, 40), -1)
    fill = int(bw * conf)
    cor = (0, 200, 0) if conf >= CONFIANCA_MINIMA else (0, 140, 255)
    cv2.rectangle(frame, (x, y), (x + fill, y + bh), cor, -1)
    cv2.putText(frame, f"Gatilho {conf:.0%}", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _desenhar_progresso_gravacao(frame, atual, total, w, h):
    """Barra de progresso da gravação de sequência."""
    progresso = int(w * min(atual, total) / total)
    cv2.rectangle(frame, (0, h - 50), (w, h), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, h - 50), (progresso, h), (0, 200, 0), -1)
    cv2.putText(frame, f"GRAVANDO {atual}/{total} frames", (20, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def _desenhar_resultado(frame, rotulo, conf, w, h):
    """Exibe resultado da classificação no centro da tela."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 2 - 220, h // 2 - 80), (w // 2 + 220, h // 2 + 80),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cor = (0, 255, 0) if conf >= CONFIANCA_MINIMA else (0, 100, 255)
    cv2.putText(frame, rotulo, (w // 2 - 160, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, cor, 6, cv2.LINE_AA)
    cv2.putText(frame, f"Confianca: {conf:.0%}", (w // 2 - 120, h // 2 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Webcam não encontrada.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = {"debug": False, "thickness": 2}

    palavra = []
    ultima_letra = [""]
    ultimo_tempo = [0]

    try:
        if USE_NEW_API:
            run_new_api(cap, state, palavra, ultima_letra, ultimo_tempo)
        else:
            run_old_api(cap, state)
    finally:
        cap.release()
        cv2.destroyAllWindows()


# 👇 FORA DO MAIN (CORRETO)
if __name__ == "__main__":
    main()