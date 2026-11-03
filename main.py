import sys
import time
import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------- Ajustes ----------------
ALTURA_TESTA = 0.40   # pontas dos dedos precisam estar acima dessa linha (0 = topo, 1 = base)
TEMPO_SEGURAR = 0.4   # segundos segurando o gesto pra ativar (evita disparo sem querer)
TEMPO_FADE = 0.8     # duração do efeito de sumir/voltar
ALTURA_DESINTEGRACAO = 0.15  # irregularidade da borda (0 a 1)
TEMPO_SUMIDO = None   # segundos até voltar sozinho (None = só volta com o gesto)
GESTO_VOLTAR = "Open_Palm"  # gesto pra reaparecer (mão aberta)
DEBUG = False         # True = desenha os pontos da mão e a linha da testa (só na prévia)
CAMERA_VIRTUAL = "/dev/video10"  # câmera virtual criada pelo v4l2loopback
# -----------------------------------------

ultimo_resultado = None


def salvar_resultado(result, output_image, timestamp_ms):
    global ultimo_resultado
    ultimo_resultado = result


def dedo_esticado(mao, ponta, junta):
    # Dedo apontando pra cima: a ponta fica acima da junta do meio
    return mao[ponta].y < mao[junta].y


def gesto_teletransporte(result):
    """Indicador + médio esticados, anelar + mindinho dobrados, pontas na altura da testa."""
    if not result or not result.hand_landmarks:
        return False
    for i, mao in enumerate(result.hand_landmarks):
        nome = result.gestures[i][0].category_name if result.gestures[i] else ""
        dois_dedos = (dedo_esticado(mao, 8, 6) and dedo_esticado(mao, 12, 10)
                      and not dedo_esticado(mao, 16, 14)
                      and not dedo_esticado(mao, 20, 18))
        na_testa = mao[8].y < ALTURA_TESTA and mao[12].y < ALTURA_TESTA
        if (nome == "Victory" or dois_dedos) and na_testa:
            return True
    return False


def tem_gesto(result, nome):
    """True se alguma mão está fazendo um dos gestos prontos do modelo."""
    if not result or not result.gestures:
        return False
    return any(g and g[0].category_name == nome for g in result.gestures)


def misturar(a, b, alpha):
    return cv2.addWeighted(a, 1 - alpha, b, alpha, 0)

def efeito_teletransporte(frame, fundo, progresso):
    h, w = frame.shape[:2]

    # Identifica a pessoa através da diferença do fundo
    diferenca = cv2.absdiff(frame, fundo)
    cinza = cv2.cvtColor(diferenca, cv2.COLOR_BGR2GRAY)

    _, mascara = cv2.threshold(cinza, 30, 255, cv2.THRESH_BINARY)

    # Suaviza a silhueta
    mascara = cv2.GaussianBlur(mascara, (7, 7), 0)

    # Desaparecimento completo
    alpha = max(0.0, 1.0 - progresso)

    # Pessoa desaparecendo
    alpha_mascara = (mascara.astype(np.float32) / 255.0) * alpha
    alpha_mascara = alpha_mascara[:, :, np.newaxis]

    saida = (frame * alpha_mascara + fundo * (1 - alpha_mascara)).astype(np.uint8)

    # Riscos horizontais
    riscos = np.zeros_like(frame)

    quantidade = 35

    for i in range(quantidade):
        y = np.random.randint(0, h)

        comprimento = int(np.random.randint(20, 120) * progresso)

        x = np.random.randint(0, w)

        x_final = min(x + comprimento, w)

        if x_final > x:
            cv2.line(
                riscos,
                (x, y),
                (x_final, y),
                (255, 255, 255),
                1
            )

    # Intensidade dos riscos
    intensidade = np.sin(progresso * np.pi)

    saida = cv2.addWeighted(
        saida, 1,
        riscos, intensidade,
        0
    )

    return saida

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=salvar_resultado)

NO_WINDOWS = sys.platform.startswith("win")

# No Windows o DirectShow abre a webcam bem mais rápido
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if NO_WINDOWS else cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: não consegui acessar a webcam")
    exit()

largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
# Linux: usa o v4l2loopback | Windows: usa a câmera virtual do OBS
cam = pyvirtualcam.Camera(width=largura, height=altura, fps=fps,
                          device=None if NO_WINDOWS else CAMERA_VIRTUAL)
print(f"Câmera virtual ativa: {cam.device} — escolha ela no Teams")

fundo = None
estado = "normal"          # normal -> sumindo -> sumido -> voltando -> normal
inicio_gesto = None
t_estado = 0.0

with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        recognizer.recognize_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(time.monotonic() * 1000))

        agora = time.monotonic()
        saida = frame.copy()
        branco = np.full_like(frame, 255)

        if fundo is None:
            pass  # sem fundo gravado ainda: só passa a imagem normal

        elif estado == "normal":
            if gesto_teletransporte(ultimo_resultado):
                if inicio_gesto is None:
                    inicio_gesto = agora
                elif agora - inicio_gesto >= TEMPO_SEGURAR:
                    estado, t_estado, inicio_gesto = "sumindo", agora, None
            else:
                inicio_gesto = None

        elif estado == "sumindo":
            p = min((agora - t_estado) / TEMPO_FADE, 1.0)

            saida = efeito_teletransporte(frame, fundo, p)

            if p >= 1:
                estado, t_estado = "sumido", agora

        elif estado == "sumido":
            saida = fundo.copy()
            # Você está invisível na tela, mas a câmera continua vendo sua mão
            if tem_gesto(ultimo_resultado, GESTO_VOLTAR):
                if inicio_gesto is None:
                    inicio_gesto = agora
                elif agora - inicio_gesto >= TEMPO_SEGURAR:
                    estado, t_estado, inicio_gesto = "voltando", agora, None
            else:
                inicio_gesto = None
            if TEMPO_SUMIDO is not None and agora - t_estado >= TEMPO_SUMIDO:
                estado, t_estado, inicio_gesto = "voltando", agora, None

        elif estado == "voltando":
            p = min((agora - t_estado) / TEMPO_FADE, 1.0)
            saida = misturar(fundo, frame, p)
            saida = misturar(saida, branco, 0.6 * (1 - p))
            if p >= 1:
                estado = "normal"

        # Manda pro Teams só o efeito, sem textos nem desenhos de ajuda
        cam.send(cv2.cvtColor(saida, cv2.COLOR_BGR2RGB))

        # Prévia só pra você (espelhada, com dicas)
        preview = saida.copy()
        if DEBUG and ultimo_resultado:
            h, w, _ = preview.shape
            cv2.line(preview, (0, int(ALTURA_TESTA * h)), (w, int(ALTURA_TESTA * h)), (255, 0, 0), 1)
            for mao in ultimo_resultado.hand_landmarks:
                for ponto in mao:
                    cv2.circle(preview, (int(ponto.x * w), int(ponto.y * h)), 3, (0, 255, 0), -1)
        preview = cv2.flip(preview, 1)
        if fundo is None:
            cv2.putText(preview, "Saia da camera e aperte B para gravar o fundo",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Teletransporte", preview)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('b'):
            fundo = frame.copy()
            estado = "normal"
            print("Fundo gravado!")
        elif tecla == ord('r'):
            estado = "normal"

cam.close()
cap.release()
cv2.destroyAllWindows()