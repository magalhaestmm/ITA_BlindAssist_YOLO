"""
FASE 1 — Detecção em tempo real com webcam (rodar no PC, não no Raspberry Pi)
=============================================================================

Objetivo desta fase:
  Ver o YOLO detectando objetos ao vivo e transformar cada detecção em uma
  "instrução navegável" (o quê + direção + distância aproximada), AINDA em
  texto no terminal. Na Fase 3, a função `anunciar()` será trocada por voz.

  >>> NESTA VERSÃO: sistema de PRIORIZAÇÃO inteligente (veja a seção abaixo).

Pré-requisitos (no seu PC):
  pip install ultralytics opencv-python

Como rodar:
  python fase1_deteccao_tempo_real.py
  Pressione 'q' na janela de vídeo para sair.
"""

import time
from collections import defaultdict
from voz import MotorDeVoz
from camera import CameraTempoReal

import cv2 as cv
from ultralytics import YOLO

# ============================================================================
# CONFIGURAÇÃO GERAL
# ============================================================================

MODELO = "yolo26n.pt"   # nano = mais rápido. Fallback: "yolo11n.pt".
CAMERA = 1              # índice da webcam.
CONFIANCA_MINIMA = 0.45
MAX_ANUNCIOS = 2         # quantos objetos "falar" por ciclo (priorização).
INTERVALO_ANUNCIO = 3.5  # segundos entre ciclos normais de anúncio.
COOLDOWN_OBJETO = 4.0    # segundos para não repetir o MESMO objeto+direção.
COOLDOWN_URGENTE = 3.0   # urgentes podem se repetir mais rápido.
MOSTRAR_JANELA = True

# Zonas horizontais (centro do objeto). "à frente" = caminho de colisão.
LIMITE_ESQUERDA = 0.40
LIMITE_DIREITA = 0.60

# Estimativa GROSSEIRA de distância pela fração da tela ocupada (calibrar!).
LIMITE_MUITO_PERTO = 0.25
LIMITE_PERTO = 0.10
LIMITE_MEDIO = 0.03

# ============================================================================
# CONFIGURAÇÃO DA PRIORIZAÇÃO  <<< O CORAÇÃO DESTE SCRIPT
# ============================================================================
#
# prioridade = PESO_CLASSE * FATOR_DIRECAO * FATOR_DISTANCIA
#
# A MULTIPLICAÇÃO é proposital: se qualquer fator for baixo, o objeto some da
# fila. Um caminhão (peso alto) longe e à lateral cai para quase zero — que é
# exatamente o que queremos. Todos os números abaixo são pontos de partida
# para você calibrar testando (idealmente com uma pessoa cega).

# --- Peso de relevância intrínseca de cada classe (0 a 10) ---
# Pense em duas perguntas: "isso pode me machucar?" e "isso me ajuda a saber
# onde estou?". Veículos têm risco alto; marcos ajudam na localização.
PESO_CLASSE = {
    # Veículos — risco de atropelamento, prioridade máxima
    "carro": 9, "ônibus": 9, "caminhão": 9, "moto": 9, "bicicleta": 8,
    # Pessoas — movem-se, você desvia delas, relevância social
    "pessoa": 6,
    # Marcos de orientação — pouco perigo, MUITO úteis para se localizar
    "semáforo": 7, "placa de pare": 6,
    # Obstáculos estáticos no nível do corpo — risco de colisão
    "cadeira": 5, "banco": 5, "sofá": 5, "mesa": 5,
    "cachorro": 5,
    # Baixa relevância para mobilidade
    "gato": 3, "mochila": 2, "bolsa": 2, "mala": 2,
    "garrafa": 1, "copo": 1,
}
PESO_PADRAO = 3   # para classes não listadas acima

# --- Fator de direção: o que está no caminho domina ---
FATOR_DIRECAO = {
    "à frente": 1.0,
    "à esquerda": 0.45,
    "à direita": 0.45,
}

# --- Fator de distância: mais perto = mais urgente ---
FATOR_DISTANCIA = {
    "muito perto": 1.0,
    "perto": 0.65,
    "à média distância": 0.35,
    "longe": 0.12,
}

# Limiar acima do qual uma detecção é tratada como URGENTE (fura a fila e
# ganha "Atenção" na frente). Ajuste conforme sentir o sistema falando demais
# ou de menos. Ex.: carro (9) à frente (1.0) muito perto (1.0) = 9.0.
LIMIAR_URGENCIA = 5.0

# ============================================================================
# Tradução das classes do COCO para português
# ============================================================================

NOMES_PT = {
    "person": "pessoa", "bicycle": "bicicleta", "car": "carro",
    "motorcycle": "moto", "bus": "ônibus", "truck": "caminhão",
    "traffic light": "semáforo", "stop sign": "placa de pare",
    "bench": "banco", "dog": "cachorro", "cat": "gato",
    "chair": "cadeira", "couch": "sofá", "dining table": "mesa",
    "backpack": "mochila", "handbag": "bolsa", "suitcase": "mala",
    "bottle": "garrafa", "cup": "copo",
}


def nome_em_pt(nome_en: str) -> str:
    return NOMES_PT.get(nome_en, nome_en)


def calcular_direcao(centro_x_norm: float) -> str:
    if centro_x_norm < LIMITE_ESQUERDA:
        return "à esquerda"
    if centro_x_norm > LIMITE_DIREITA:
        return "à direita"
    return "à frente"


def calcular_distancia(fracao_area: float) -> str:
    if fracao_area > LIMITE_MUITO_PERTO:
        return "muito perto"
    if fracao_area > LIMITE_PERTO:
        return "perto"
    if fracao_area > LIMITE_MEDIO:
        return "à média distância"
    return "longe"


def calcular_prioridade(nome: str, direcao: str, distancia: str) -> float:
    """
    O coração do sistema. Combina O QUÊ + ONDE + QUÃO PERTO num único número.
    Multiplicação (não soma): qualquer fator baixo derruba a prioridade.
    """
    peso = PESO_CLASSE.get(nome, PESO_PADRAO)
    fd = FATOR_DIRECAO.get(direcao, 0.45)
    fdist = FATOR_DISTANCIA.get(distancia, 0.12)
    return peso * fd * fdist


def anunciar(frase: str, urgente: bool = False, motor = None) -> None:
    """
    Saída para o usuário. FASE 1: imprime. FASE 3: troque por voz (pyttsx3).
    Itens urgentes ganham um prefixo de alerta.
    """
    prefixo = "⚠️  ATENÇÃO: " if urgente else "🔊 "
    print(f"  {prefixo}{frase}")

    if motor:
        motor.falar(frase, limpar_fila=urgente)


def main() -> None:
    print(f"Carregando modelo {MODELO}...")
    modelo = YOLO(MODELO)

    motor_voz = MotorDeVoz()

    cap = CameraTempoReal(CAMERA)

    if not cap.isOpened():
        print(f"ERRO: não consegui abrir a câmera {CAMERA}.")
        return

    print("Câmera aberta. Pressione 'q' para sair.\n")

    ultimo_anuncio = defaultdict(lambda: 0.0)
    ultimo_ciclo = 0.0
    t_anterior = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERRO: falha ao ler o quadro da câmera.")
            break

        altura, largura = frame.shape[:2]
        area_total = float(altura * largura)

        resultados = modelo(frame, conf=CONFIANCA_MINIMA, verbose=False)
        boxes = resultados[0].boxes

        deteccoes = []
        for box in boxes:
            nome = nome_em_pt(modelo.names[int(box.cls)])
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            centro_x_norm = ((x1 + x2) / 2) / largura
            fracao_area = ((x2 - x1) * (y2 - y1)) / area_total

            direcao = calcular_direcao(centro_x_norm)
            distancia = calcular_distancia(fracao_area)
            prioridade = calcular_prioridade(nome, direcao, distancia)

            deteccoes.append({
                "nome": nome,
                "conf": conf,
                "direcao": direcao,
                "distancia": distancia,
                "prioridade": prioridade,
                "urgente": prioridade >= LIMIAR_URGENCIA,
                "caixa": (int(x1), int(y1), int(x2), int(y2)),
            })

        # PRIORIZAÇÃO: ordena pela prioridade calculada (não mais só pela área).
        deteccoes.sort(key=lambda d: d["prioridade"], reverse=True)

        agora = time.time()
        urgentes = [d for d in deteccoes if d["urgente"]]

        # 1) URGENTES furam a fila e são anunciados imediatamente, sempre que
        #    passou o cooldown curto. A segurança vence o ritmo normal.
        for d in urgentes:
            chave = (d["nome"], d["direcao"])
            if agora - ultimo_anuncio[chave] >= COOLDOWN_URGENTE:
                anunciar(f"{d['nome']} {d['direcao']}, {d['distancia']}", urgente=True, motor = motor_voz)
                ultimo_anuncio[chave] = agora

        # 2) NÃO-URGENTES seguem o ritmo calmo, no máximo MAX_ANUNCIOS por ciclo.
        if agora - ultimo_ciclo >= INTERVALO_ANUNCIO and deteccoes:
            ultimo_ciclo = agora
            anunciados = 0
            for d in deteccoes:
                if d["urgente"]:
                    continue  # já tratado acima
                if anunciados >= MAX_ANUNCIOS:
                    break
                chave = (d["nome"], d["direcao"])
                if agora - ultimo_anuncio[chave] >= COOLDOWN_OBJETO:
                    anunciar(f"{d['nome']} {d['direcao']}, {d['distancia']}")
                    ultimo_anuncio[chave] = agora
                    anunciados += 1

        # --- Desenho na tela (só para você; o usuário cego não vê) ---
        if MOSTRAR_JANELA:
            for d in deteccoes:
                x1, y1, x2, y2 = d["caixa"]
                cor = (0, 0, 255) if d["urgente"] else (0, 200, 0)
                rotulo = f"{d['nome']} {d['direcao']} p={d['prioridade']:.1f}"
                cv.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv.putText(frame, rotulo, (x1, max(20, y1 - 8)),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

            cv.line(frame, (int(LIMITE_ESQUERDA * largura), 0),
                     (int(LIMITE_ESQUERDA * largura), altura), (80, 80, 80), 1)
            cv.line(frame, (int(LIMITE_DIREITA * largura), 0),
                     (int(LIMITE_DIREITA * largura), altura), (80, 80, 80), 1)

            t_agora = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, t_agora - t_anterior))
            t_anterior = t_agora
            cv.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv.imshow("Fase 1 - Priorizacao (q para sair)", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()
    motor_voz.encerrar()
    print("\nEncerrado.")


if __name__ == "__main__":
    main()