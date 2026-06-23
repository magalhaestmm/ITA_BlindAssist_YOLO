import pyttsx3
import threading
import queue

class MotorDeVoz:
    def __init__(self, volume=1.0, velocidade=140):
        """
        Inicializa o motor TTS em uma thread separada para não travar o OpenCV.
        """
        self.engine = pyttsx3.init()
        
        # Configurações de voz
        self.engine.setProperty('volume', volume)
        self.engine.setProperty('rate', velocidade) # Velocidade da fala (palavras por minuto)
        
        # Tenta forçar a voz para português, caso o sistema tenha várias instaladas
        vozes = self.engine.getProperty('voices')
        for voz in vozes:
            if 'pt' in voz.languages or 'brazil' in voz.name.lower():
                self.engine.setProperty('voice', voz.id)
                break

        # Sistema de fila e controle da Thread
        self.fila_de_falas = queue.Queue()
        self.rodando = True
        
        # Inicia a thread que vai rodar em paralelo ao seu while True da câmera
        self.thread_voz = threading.Thread(target=self._processar_fila, daemon=True)
        self.thread_voz.start()
        print("[VOZ] Motor de áudio inicializado com sucesso.")

    def _processar_fila(self):
        """
        Loop interno da thread. Fica esperando itens na fila e fala um por um.
        """
        while self.rodando:
            # Pega a próxima frase da fila (bloqueia até ter algo)
            frase = self.fila_de_falas.get()
            
            # Se receber None, é o sinal para desligar
            if frase is None:
                break
                
            self.engine.say(frase)
            self.engine.runAndWait()
            self.fila_de_falas.task_done()

    def falar(self, frase: str, limpar_fila: bool = False):
        """
        Adiciona uma nova frase para ser falada.
        Se limpar_fila=True, cancela o que ia dizer para falar algo urgente.
        """
        if limpar_fila:
            # Esvazia a fila atual de forma bruta
            with self.fila_de_falas.mutex:
                self.fila_de_falas.queue.clear()
                
        self.fila_de_falas.put(frase)

    def encerrar(self):
        """
        Desliga a thread de forma segura.
        """
        self.rodando = False
        self.fila_de_falas.put(None) # Manda o sinal de parada
        self.thread_voz.join()