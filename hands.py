import cv2
import mediapipe as mp

# Inicializa a captura de vídeo da câmera
video = cv2.VideoCapture(0)

hands = mp.solutions.hands # Importa o módulo de detecção de mãos do mediapipe
Hands = hands.Hands(max_num_hands=1) # Cria uma instância do detector de mãos
mpDwaw = mp.solutions.drawing_utils # Módulo para desenhar landmarks das mãos na imagem

# Loop para processar cada frame do vídeo
while True:
    # Fazer leitura do frame de video
    check,img = video.read()

    # Converter a imagem de BGR para RGB
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = Hands.process(imgRGB) # Processar a img para detectar as mãos
    handsPoints = results.multi_hand_landmarks # Obter as landmarks das mãos detectadas
    
    # Obter a altura e largura da imagem
    h,w,_ = img.shape 

    pontos = [] #Variável para armazenar as coordenadas dos pontos das mãos
    
    # Verificar se as mãos foram detectadas
    if handsPoints:
        for points in handsPoints:
            # Desenhar as landmarks das mãos na imagem
            mpDwaw.draw_landmarks(img, points,hands.HAND_CONNECTIONS)

            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)

                # Armazenar as coordenadas (x, y) dos pontos das mãos
                pontos.append((cx,cy)) 
            
            # Pontos de referência dos dedos
            dedos = [8,12,16,20]

            # Contador para contar os dedos levantados
            contador = 0
            
            if pontos:
                # Verificar se o polegar está levantado
                if pontos[4][0] < pontos[3][0]:
                    contador += 1
                for x in dedos:
                   # Verificar se os outros dedos estão levantados
                   if pontos[x][1] < pontos[x-2][1]:
                       contador +=1

            # Exibir o contador
            cv2.putText(img,str(contador),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)
            #print(contador)

    # Exibe a imagem com as landmarks e o contador
    cv2.imshow('Imagem',img)
    cv2.waitKey(1)

    # Tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o objeto de captura de vídeo e fecha as janelas
video.release()
cv2.destroyAllWindows()