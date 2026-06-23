import cv2 as cv
import threading

class CameraTempoReal:
    def __init__(self, indice_camera=0):
        """
        Inicia a câmera e uma thread paralela que mantém apenas o frame mais recente,
        eliminando o delay de buffer do OpenCV.
        """
        self.cap = cv.VideoCapture(indice_camera)
        
        # Otimização de resolução nativa para acelerar o USB
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Lê o primeiro frame para ter algo na memória
        self.ok, self.frame = self.cap.read()
        self.rodando = True
        
        # Inicia o aspirador de frames em segundo plano
        self.thread = threading.Thread(target=self._atualizar_frame, daemon=True)
        self.thread.start()

    def _atualizar_frame(self):
        """
        Loop infinito que roda o mais rápido que a porta USB aguentar.
        Se chegar frame novo, ele esmaga o frame antigo.
        """
        while self.rodando:
            ok, frame = self.cap.read()
            if ok:
                self.ok = ok
                self.frame = frame

    def isOpened(self):
        """
        Retorna se a captura de vídeo foi inicializada com sucesso.
        """
        return self.cap.isOpened()

    def read(self):
        """
        Substitui o comportamento padrão do OpenCV para entregar sempre a foto MAIS NOVA.
        """
        return self.ok, self.frame

    def release(self):
        """
        Limpa a bagunça ao fechar o programa.
        """
        self.rodando = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()