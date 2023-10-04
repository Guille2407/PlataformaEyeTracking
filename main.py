from operator import rshift
from queue import Queue
from pynput.mouse import Button, Controller
from tkinter import *
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import mediapipe as mp
import imutils
from threading import Thread, Condition
import math
import time
import webview
# Para ver la face mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
# Para controlar mouse via pynput
mouse = Controller()
mp_face_mesh = mp.solutions.face_mesh
# Indices del ojo izquierdo (mesh)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
# Indices del ojo derecho (mesh)
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_EYE_TOP = 223
RIGHT_EYE_BOTTOM = 230
RIGHT_EYE_INT = 133
RIGHT_EYE_EXT = 33
LEFT_EYE_TOP = 443
LEFT_EYE_BOTTOM = 450
LEFT_EYE_INT = 362
LEFT_EYE_EXT = 263
RIGHT_EYE_EYELID_TOP = 160
RIGHT_EYE_EYELID_BOT = 144
LEFT_EYE_EYELID_TOP = 387
LEFT_EYE_EYELID_BOT = 373
MOUTH_TOP = 13 
MOUTH_BOTTOM = 14
PARPADEOS = 0
font = cv.FONT_HERSHEY_SIMPLEX
cantidad = 1
ini = True
cola = []
construccion = []


def visualizar(canva):
    global pantalla, frame, ini, PARPADEOS
    misComandos = ["arriba", "abajo", "derecha", "izquierda"]
    comandosReales = ["forward", "backward", "turn-right", "turn-left"]
    construccionX = 250
    construccionY = 620
    # Leer la videocaptura
    tiempoPasado = 0.0
    seleccion = 0
    while(ini):
        tiempoActual = time.time() 
        with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
        ) as face_mesh:
            if cap is not None:
                ret, frame = cap.read()
            # Si es correcta
            if ret == True:
                
                frame = cv.flip(frame, 1)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img_h, img_w = frame.shape[:2]
                results = face_mesh.process(rgb_frame)

                # Draw the face mesh annotations on the image.
                frame.flags.writeable = True

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:

                        #mp_drawing.draw_landmarks(
                        #image=frame,
                        #landmark_list=face_landmarks,
                        #connections=mp_face_mesh.FACEMESH_TESSELATION,
                        #landmark_drawing_spec=None,
                        #connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        #mp_drawing.draw_landmarks(
                        #image=frame,
                        #landmark_list=face_landmarks,
                        #connections=mp_face_mesh.FACEMESH_CONTOURS,
                        #landmark_drawing_spec=None,
                        #connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        #mp_drawing.draw_landmarks(
                        #image=frame,
                        #landmark_list=face_landmarks,
                        #connections=mp_face_mesh.FACEMESH_IRISES,
                        #landmark_drawing_spec=None,
                        #connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(
                        int) for p in results.multi_face_landmarks[0].landmark])
                        

                    #Marcadores de ojos
                    cv.circle(frame, mesh_points[RIGHT_EYE_EXT], radius=2, color=(255, 255, 255), thickness=-1)
                    cv.circle(frame, mesh_points[RIGHT_EYE_INT], radius=2, color=(255, 255, 255), thickness=-1)
                    cv.circle(frame, mesh_points[RIGHT_EYE_TOP], radius=2, color=(255, 255, 255), thickness=-1)
                    cv.circle(frame, mesh_points[RIGHT_EYE_BOTTOM], radius=2, color=(255, 255, 255), thickness=-1)
                    cv.circle(frame, mesh_points[LEFT_EYE_EXT], radius=2, color=(255, 255, 255), thickness=-1)
                    cv.circle(frame, mesh_points[LEFT_EYE_INT], radius=2, color=(255, 255, 255), thickness=-1)   
                    cv.circle(frame, mesh_points[LEFT_EYE_TOP], radius=2, color=(255, 255, 255), thickness=-1)
                    cv.circle(frame, mesh_points[LEFT_EYE_BOTTOM], radius=2, color=(255, 255, 255), thickness=-1)       
                    #( 3, 95, 255)
                    #Marcadores de pupilas    
                    (l_cx, l_cy), l_radius = cv.minEnclosingCircle(
                        mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
                        mesh_points[RIGHT_IRIS])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    cv.circle(frame, center_left, int(l_radius),
                        (255, 0, 255), 3, cv.LINE_AA)
                    cv.circle(frame, center_right, int(r_radius),
                        (255, 0, 255), 3, cv.LINE_AA)
                    #Marcadores centro de pupila    
                    cv.circle(frame, center_left, radius=2, color=(255, 255, 255), thickness=-1)
                    cv.circle(frame, center_right, radius=2, color=(255, 255, 255), thickness=-1)
                    #Marcadores boca
                    cv.circle(frame, mesh_points[MOUTH_TOP], radius=2, color=(255, 255, 255), thickness=-1)#color=(50, 168, 82), thickness=-1)
                    cv.circle(frame, mesh_points[MOUTH_BOTTOM], radius=2, color=(255, 255, 255), thickness=-1)#color=(50, 168, 82), thickness=-1)

                    distOjoIzqH = euclaideanDistance(mesh_points[LEFT_EYE_INT],mesh_points[LEFT_EYE_EXT])
                    distOjoDerH = euclaideanDistance(mesh_points[RIGHT_EYE_INT],mesh_points[RIGHT_EYE_EXT])
                    distOjoIzqV = euclaideanDistance(mesh_points[LEFT_EYE_TOP],mesh_points[LEFT_EYE_BOTTOM])
                    distOjoDerV = euclaideanDistance(mesh_points[RIGHT_EYE_TOP],mesh_points[RIGHT_EYE_BOTTOM])

                    #Trigger Boca
                    distBoca = euclaideanDistance(mesh_points[MOUTH_TOP], mesh_points[MOUTH_BOTTOM])

                    #Parpadeo
                    disParpados = (euclaideanDistance(mesh_points[RIGHT_EYE_EYELID_BOT], mesh_points[RIGHT_EYE_EYELID_TOP])+euclaideanDistance(mesh_points[LEFT_EYE_EYELID_BOT], mesh_points[LEFT_EYE_EYELID_TOP]))/2
                    ratioParpadeo = disParpados / distOjoIzqH

                    ratioIzq = (((distOjoDerH + distOjoIzqH)/2) / (euclaideanDistance(mesh_points[LEFT_EYE_INT], center_left) + euclaideanDistance(mesh_points[RIGHT_EYE_EXT], center_right))/2)
                    ratioDer = (((distOjoDerH + distOjoIzqH)/2) / (euclaideanDistance(mesh_points[LEFT_EYE_EXT], center_left) + euclaideanDistance(mesh_points[RIGHT_EYE_INT], center_right))/2)
                    ratioArriba = (((distOjoDerV + distOjoIzqV)/2) / (euclaideanDistance(mesh_points[LEFT_EYE_TOP], center_left) + euclaideanDistance(mesh_points[RIGHT_EYE_TOP], center_right))/2)
                    ratioAbajo = (((distOjoDerV + distOjoIzqV)/2) / (euclaideanDistance(mesh_points[LEFT_EYE_BOTTOM], center_left) + euclaideanDistance(mesh_points[RIGHT_EYE_BOTTOM], center_right))/2)

                    if(ratioParpadeo < 0.08):
                        PARPADEOS = PARPADEOS + 1

                    tiempoFin = time.time()
                    tiempoPasado = tiempoPasado + (tiempoFin - tiempoActual)
                    #print("tiempo de ", tiempoPasado)
                    if(tiempoPasado > 3): #3
                        if(distBoca > 2): #2
                                if(ratioIzq > sliderOjoIzq.get()): #0.55
                                    #Hacer algo con esta señalq
                                    cv.putText(frame, 
                                    'Izquierda', 
                                    (50, 50), 
                                    font, 1, 
                                    (0, 255, 255), 
                                    2, 
                                    cv.LINE_4)
                                    #seleccion = actualizarSeleccion(seleccion, -1)
                                    seleccion = mover_Selector(canva, seleccion,-1)
                                    canvas.itemconfigure(texto, text=misComandos[seleccion])
                                    tiempoPasado = 0.0
                                    #mouse.move(-10,0)
                                if(ratioDer > sliderOjoDer.get()): #0.55
                                    cv.putText(frame, 
                                    'Derecha', 
                                    (50, 70), 
                                    font, 1, 
                                    (3, 95, 255), 
                                    2, 
                                    cv.LINE_4)
                                    seleccion = mover_Selector(canva, seleccion, 1)
                                    tiempoPasado = 0.0
                                    canvas.itemconfigure(texto, text=misComandos[seleccion])
                                    #mouse.move(10,0)
                                if(ratioArriba > sliderOjoArriba.get()): #0.5
                                    cv.putText(frame, 
                                    'Arriba', 
                                    (50, 90), 
                                    font, 1, 
                                    (3, 20, 255), 
                                    2, 
                                    cv.LINE_4)
                                    #mouse.move(0,-10)    
                                if(ratioAbajo > sliderOjoAbajo.get()): #0.55
                                    cv.putText(frame, 
                                    'Abajo', 
                                    (50, 90), 
                                    font, 1, 
                                    (3, 200, 255), 
                                    2, 
                                    cv.LINE_4)  
                                    #mouse.move(0,10)     
                    if(PARPADEOS > 30):
                        PARPADEOS = 0   
                        valor = canvas.itemcget(resultado, 'text')
                        #canvas.itemconfigure(resultado, text= valor +" "+ misComandos[seleccion])
                        comando = misComandos[seleccion]
                        if comando == "arriba":
                            construccion.append(canvas.create_image(construccionX, construccionY, anchor=NW, image=imagenPic1))
                            construccionX = construccionX + 100
                        elif comando == "abajo":
                            construccion.append(canvas.create_image(construccionX, construccionY, anchor=NW, image=imagenPic3))
                            construccionX = construccionX + 100
                        elif comando == "derecha":
                            construccion.append(canvas.create_image(construccionX, construccionY, anchor=NW, image=imagenPic2))
                            construccionX = construccionX + 100
                        elif comando == "izquierda":
                            construccion.append(canvas.create_image(construccionX, construccionY, anchor=NW, image=imagenPic4))
                            construccionX = construccionX + 100
                        cola.append(comandosReales[seleccion])
                # Rendimensionamos el video
                frame = imutils.resize(frame, width=320)
                
                # Convertimos el video
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)

                # Mostramos en el GUI
                lblVideo.configure(image=img)
                lblVideo.image = img

            else:
                cap.release()

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def iniciar(th1):
    global cap
    # Elegimos la camara
    #print(texto)
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    print("Inciar")
    th1.start()

def finalizar():
    global iniciar
    ini = False
    cap.release()
    cv.destroyAllWindows()

def mover_Selector(canvas, pos, direccion):
    if(direccion == 1):
        pos = pos + 1
        if(pos > 3 ):
            canvas.moveto(selector, 550, 300)
            pos = 0
        else:
            canvas.move(selector, 120, 0)
    else:
        pos = pos - 1
        if(pos < 0 ):
            canvas.moveto(selector, 910, 300)
            pos = 3
        else:
            canvas.move(selector, -120, 0)
    return pos 

def salir():
    exit()

def limpiarTexto(canva):
    #canvas.itemconfigure(resultado, text="")
    cola.clear()
    for x in construccion:
        x.delete()
    

def nuevaVentana():
    url = ""
    for i in cola:
        url = url +"{"+"\"id\""+":"+str(cola.index(i))+","+"\"move\""+":"+"\""+i+"\""+"},"
    url = url[0 :len(url)-1]
    print(url)
    webview.create_window('Ejecución','https://incuba.fi.uncoma.edu.ar/kit/app-tangible/run-word/?programLI={"moves":['+url+'],"mundo":"word1"}',
                          x=w-800, y= h-1020, width=800, height= 620)
    webview.start()


# Pantalla de Tk
pantalla = Tk()
pantalla.title("Ambiente Eye Tracking")
pantalla.geometry("1280x720")
pantalla.state('zoomed')
w, h = pantalla.winfo_screenwidth(), pantalla.winfo_screenheight()
canvas = Canvas(pantalla)
imagenFlecha1 = PhotoImage(file="imagenes/f_arriba.png")
imagenFlecha2 = PhotoImage(file="imagenes/f_derecha.png")
imagenFlecha3 = PhotoImage(file="imagenes/f_abajo.png")
imagenFlecha4 = PhotoImage(file="imagenes/f_izquierda.png")
imagenPic1 = PhotoImage(file="imagenes/pic_arriba.png")
imagenPic2 = PhotoImage(file="imagenes/pic_derecha.png")
imagenPic3 = PhotoImage(file="imagenes/pic_abajo.png")
imagenPic4 = PhotoImage(file="imagenes/pic_izquierda.png")
imagenSelector = PhotoImage(file="imagenes/Nofondo.png")
imagenFondoCamara = PhotoImage(file="imagenes/fondoCamara.png")
flechaArriba = canvas.create_image(550, 300, anchor=NW, image=imagenFlecha1)
flechaAbajo = canvas.create_image(670, 300, anchor=NW, image=imagenFlecha3)
flechaDer = canvas.create_image(790, 300, anchor=NW, image=imagenFlecha2)
flechaIzq = canvas.create_image(910, 300, anchor=NW, image=imagenFlecha4)
selector = canvas.create_image(550, 300, anchor=NW, image=imagenSelector)
fondoCamara = canvas.create_image(w-500, h-405, anchor=NW, image=imagenFondoCamara)
texto = canvas.create_text(550, 500, anchor=NW , text="Arriba", font=('arial', 50, 'bold'))
textoSec = canvas.create_text(5, 620, anchor=NW , text="Secuencia: ", font=('arial', 30, 'bold'))
textoVideo = canvas.create_text(w-370, h-450, anchor=NW , text="Video en vivo", font=('arial', 30, 'bold'))
resultado = canvas.create_text(300, 600, anchor=NW , text="", font=('arial', 50, 'bold'), fill='green')

areaOtrosSliders = canvas.create_rectangle(500,800, 770, 1000,outline='black', fill='#cddef1')
areaSensibilidadDerIzq = canvas.create_rectangle(800,800, 1070, 1000,outline='black', fill='#cddef1')
areaSensibilidadArriAbaj = canvas.create_rectangle(1100,800, 1370, 1000,outline='black', fill='#cddef1')

canvas.pack(fill="both", expand=True)

# Botones

# Iniciar Video
imagenBI = PhotoImage(file="imagenes/iniciar.png")
inicio = Button(pantalla, text="Iniciar", image=imagenBI, height="50", width="150", command=lambda: iniciar(t1))
inicio.place(x = 20, y = h-130)

# Finalizar Video
imagenBF = PhotoImage(file="imagenes/ajustes.png")
fin = Button(pantalla, text="Detener", image=imagenBF, height="50", width="150", command=finalizar)
fin.place(x = 20, y = h-200)

# Abrir Nueva ventana
imagenBE = PhotoImage(file="imagenes/ejecutar.png")
ejecutar = Button(pantalla, text="Ejecutar", image=imagenBE, height="50", width="150", command=lambda: nuevaVentana())
ejecutar.place(x = 20, y = h-270)

# Salir
imagenBS = PhotoImage(file="imagenes/salir.png")
fin = Button(pantalla, text="Salir", image=imagenBS, height="50", width="150", command=salir)
fin.place(x = 200, y = h-130)

# Limpiar
imagenBL = PhotoImage(file="imagenes/limpiar.png")
inicio = Button(pantalla, text="Limpiar", image=imagenBL, height="50", width="150", command=lambda: limpiarTexto(canvas))
inicio.place(x = 200, y = h-200)

# Sliders
sliderOjoCerrado = Scale(canvas, highlightbackground='black', relief=SUNKEN, length = 230, font= ('Helvetica bold', 11,'bold'),
                        label="Sensibilidad ojo cerrado",  bg='#6e9bd2', fg='black',
                          activebackground='#9d9bde' , from_ = 0.0, to = 1.0, resolution=0.01, orient=HORIZONTAL)
sliderOjoCerrado.place(x = 517, y = h-250)
sliderOjoCerrado.set(0.08)

sliderTiempoSeleccion = Scale(canvas, highlightbackground='black', relief=SUNKEN, length = 230, font= ('Helvetica bold', 11,'bold'),
                        label="Tiempo de seleccion",  bg='#6e9bd2', fg='black',
                          activebackground='#9d9bde' , from_ = 0, to = 100, resolution=1, orient=HORIZONTAL)
sliderTiempoSeleccion.place(x = 517, y = h-170)
sliderTiempoSeleccion.set(30)

sliderOjoDer = Scale(canvas, highlightbackground='black', relief=SUNKEN, length = 230, font= ('Helvetica bold', 11,'bold'),
                        label="Sensibilidad mirar derecha", bg='#6e9bd2', fg='black',
                          activebackground='#9d9bde' , from_ = 0.0, to = 1.0, resolution=0.01, orient=HORIZONTAL)
sliderOjoDer.place(x = 817, y = h-250) #(x = w-650, y = h-300)
sliderOjoDer.set(0.55)

sliderOjoIzq = Scale(canvas, highlightbackground='black', relief=SUNKEN,  length = 230, font= ('Helvetica bold', 11,'bold'),
                        label="Sensibilidad mirar izquierda",  bg='#6e9bd2', fg='black',
                          activebackground='#9d9bde' , from_ = 0.0, to = 1.0, resolution=0.01, orient=HORIZONTAL)
sliderOjoIzq.place(x = 817, y = h-170)
sliderOjoIzq.set(0.55)

sliderOjoArriba = Scale(canvas, highlightbackground='black', relief=SUNKEN, length = 230, font= ('Helvetica bold', 11,'bold'),
                        label="Sensibilidad para mirar arriba",  bg='#6e9bd2', fg='black',
                          activebackground='#9d9bde' , from_ = 0.0, to = 1.0, resolution=0.01, orient=HORIZONTAL)
sliderOjoArriba.place(x = 1117, y = h-250)
sliderOjoArriba.set(0.5)

sliderOjoAbajo = Scale(canvas, highlightbackground='black', relief=SUNKEN, length = 230, font= ('Helvetica bold', 11,'bold'),
                        label="Sensibilidad para mirar abajo", bg='#6e9bd2', fg='black',
                          activebackground='#9d9bde' , from_ = 0.0, to = 1.0, resolution=0.01, orient=HORIZONTAL)
sliderOjoAbajo.place(x = 1117, y = h-170)
sliderOjoAbajo.set(0.55)

# Video
lblVideo = Label(pantalla)
lblVideo.place(x = w-420, y = h-350)

# Hilos
t1 = Thread(target=visualizar, args=[canvas])
pantalla.mainloop()

