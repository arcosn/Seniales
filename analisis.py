import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import convolve
from scipy.fft import fft, fftfreq
import os

os.makedirs("resultados", exist_ok=True)

imagen = None
video_ruta = ""

audio1 = None
audio2 = None
audio_conv = None
fs1 = None
fs2 = None
fs_conv = None


# ==========================
# IMAGEN
# ==========================

def mostrar_imagen(img, titulo):
    cv2.imshow(titulo, img)
    cv2.imwrite(f"resultados/{titulo}.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cargar_imagen():
    global imagen

    ruta = filedialog.askopenfilename(
        filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
    )

    if ruta:
        imagen = cv2.imread(ruta)

        if imagen is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen.")
            return

        mostrar_imagen(imagen, "Imagen Original")


def aplicar_filtro():
    global imagen

    if imagen is None:
        messagebox.showerror("Error", "Primero carga una imagen.")
        return

    filtro = filtro_var.get()
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    if filtro == "Roberts":
        kx = np.array([[1, 0], [0, -1]])
        ky = np.array([[0, 1], [-1, 0]])

        x = cv2.filter2D(gris, -1, kx)
        y = cv2.filter2D(gris, -1, ky)
        resultado = cv2.addWeighted(x, 0.5, y, 0.5, 0)

    elif filtro == "Prewitt":
        kx = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])

        ky = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ])

        x = cv2.filter2D(gris, -1, kx)
        y = cv2.filter2D(gris, -1, ky)
        resultado = cv2.addWeighted(x, 0.5, y, 0.5, 0)

    elif filtro == "Sobel horizontal":
        resultado = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
        resultado = cv2.convertScaleAbs(resultado)

    elif filtro == "Sobel vertical":
        resultado = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
        resultado = cv2.convertScaleAbs(resultado)

    elif filtro == "Sobel ambos":
        sx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)

        sx = cv2.convertScaleAbs(sx)
        sy = cv2.convertScaleAbs(sy)

        resultado = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)

    elif filtro == "Laplace":
        resultado = cv2.Laplacian(gris, cv2.CV_64F)
        resultado = cv2.convertScaleAbs(resultado)

    elif filtro == "LOG":
        suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
        resultado = cv2.Laplacian(suavizada, cv2.CV_64F)
        resultado = cv2.convertScaleAbs(resultado)

    else:
        messagebox.showerror("Error", "Filtro no válido.")
        return

    mostrar_imagen(resultado, filtro)


# ==========================
# AUDIO
# ==========================

def preparar_audio(audio):
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    audio = audio.astype(np.float32)

    maximo = np.max(np.abs(audio))
    if maximo != 0:
        audio = audio / maximo

    return audio


def cargar_audio_1():
    global audio1, fs1

    ruta = filedialog.askopenfilename(
        filetypes=[("Audio WAV", "*.wav")]
    )

    if ruta:
        audio1, fs1 = sf.read(ruta)
        audio1 = preparar_audio(audio1)

        messagebox.showinfo(
            "Audio 1",
            f"Audio 1 cargado correctamente.\nFrecuencia: {fs1} Hz"
        )


def cargar_audio_2():
    global audio2, fs2

    ruta = filedialog.askopenfilename(
        filetypes=[("Audio WAV", "*.wav")]
    )

    if ruta:
        audio2, fs2 = sf.read(ruta)
        audio2 = preparar_audio(audio2)

        messagebox.showinfo(
            "Audio 2",
            f"Audio 2 cargado correctamente.\nFrecuencia: {fs2} Hz"
        )


def reproducir_audio_1():
    if audio1 is None:
        messagebox.showerror("Error", "Primero carga el audio 1.")
        return

    sd.play(audio1, fs1)


def reproducir_audio_2():
    if audio2 is None:
        messagebox.showerror("Error", "Primero carga el audio 2.")
        return

    sd.play(audio2, fs2)


def reproducir_audio_convolucionado():
    if audio_conv is None:
        messagebox.showerror("Error", "Primero genera la convolución.")
        return

    sd.play(audio_conv, fs_conv)


def detener_audio():
    sd.stop()


def graficar_senal(audio, fs, titulo, archivo):
    tiempo = np.arange(len(audio)) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(tiempo, audio)
    plt.title(titulo)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.savefig(f"resultados/{archivo}.png")
    plt.show()


def graficar_fft(audio, fs, titulo, archivo):
    n = len(audio)

    yf = fft(audio)
    xf = fftfreq(n, 1 / fs)

    positivos = xf >= 0

    plt.figure(figsize=(10, 4))
    plt.plot(xf[positivos], np.abs(yf[positivos]))
    plt.xlim(0, 4000)
    plt.title(titulo)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.grid()
    plt.savefig(f"resultados/{archivo}.png")
    plt.show()


def mostrar_senal_audio_1():
    if audio1 is None:
        messagebox.showerror("Error", "Primero carga el audio 1.")
        return

    graficar_senal(audio1, fs1, "Señal en el tiempo - Audio 1", "senal_audio_1")


def mostrar_senal_audio_2():
    if audio2 is None:
        messagebox.showerror("Error", "Primero carga el audio 2.")
        return

    graficar_senal(audio2, fs2, "Señal en el tiempo - Audio 2", "senal_audio_2")


def mostrar_senal_convolucionada():
    if audio_conv is None:
        messagebox.showerror("Error", "Primero genera la convolución.")
        return

    graficar_senal(
        audio_conv,
        fs_conv,
        "Señal en el tiempo - Audio convolucionado",
        "senal_audio_convolucionado"
    )


def mostrar_fft_audio_1():
    if audio1 is None:
        messagebox.showerror("Error", "Primero carga el audio 1.")
        return

    graficar_fft(audio1, fs1, "Espectro de frecuencia - Audio 1", "fft_audio_1")


def mostrar_fft_audio_2():
    if audio2 is None:
        messagebox.showerror("Error", "Primero carga el audio 2.")
        return

    graficar_fft(audio2, fs2, "Espectro de frecuencia - Audio 2", "fft_audio_2")


def mostrar_fft_convolucionada():
    if audio_conv is None:
        messagebox.showerror("Error", "Primero genera la convolución.")
        return

    graficar_fft(
        audio_conv,
        fs_conv,
        "Espectro de frecuencia - Audio convolucionado",
        "fft_audio_convolucionado"
    )


def convolucionar_audios():
    global audio_conv, fs_conv

    if audio1 is None or audio2 is None:
        messagebox.showerror("Error", "Debes cargar el audio 1 y el audio 2.")
        return

    if fs1 != fs2:
        messagebox.showerror(
            "Error",
            "Los dos audios deben tener la misma frecuencia de muestreo."
        )
        return

    fs_conv = fs1

    max_len = min(len(audio1), len(audio2))
    a1 = audio1[:max_len]
    a2 = audio2[:max_len]

    audio_conv = convolve(a1, a2, mode="same")

    maximo = np.max(np.abs(audio_conv))
    if maximo != 0:
        audio_conv = audio_conv / maximo

    sf.write("resultados/audio_convolucionado.wav", audio_conv, fs_conv)

    messagebox.showinfo(
        "Convolución",
        "Audio convolucionado generado correctamente.\nSe guardó en resultados/audio_convolucionado.wav"
    )


# ==========================
# VIDEO
# ==========================

def cargar_video():
    global video_ruta

    video_ruta = filedialog.askopenfilename(
        filetypes=[("Videos", "*.mp4 *.avi *.mov")]
    )

    if video_ruta:
        messagebox.showinfo("Video", "Video cargado correctamente.")


def procesar_video():
    if video_ruta == "":
        messagebox.showerror("Error", "Primero carga un video.")
        return

    cap = cv2.VideoCapture(video_ruta)

    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo abrir el video.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)

        sx = cv2.convertScaleAbs(sx)
        sy = cv2.convertScaleAbs(sy)

        sobel = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)

        cv2.imshow("Video Original", frame)
        cv2.imshow("Video Sobel", sobel)

        if cv2.waitKey(25) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================
# INTERFAZ
# ==========================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ventana = ctk.CTk()
ventana.title("Procesamiento de señales multimedia")
ventana.geometry("950x820")
ventana.resizable(False, False)

titulo = ctk.CTkLabel(
    ventana,
    text="Proyecto de Procesamiento de Señales",
    font=("Arial", 28, "bold")
)
titulo.pack(pady=20)

subtitulo = ctk.CTkLabel(
    ventana,
    text="Procesamiento de imagen, audio y video",
    font=("Arial", 15)
)
subtitulo.pack(pady=(0, 10))

contenedor = ctk.CTkScrollableFrame(
    ventana,
    width=880,
    height=640,
    corner_radius=18
)
contenedor.pack(padx=25, pady=10, fill="both", expand=True)


# ==========================
# SECCIÓN IMAGEN
# ==========================

frame_img = ctk.CTkFrame(contenedor, corner_radius=16)
frame_img.pack(padx=20, pady=15, fill="x")

ctk.CTkLabel(
    frame_img,
    text="Imagen / Fotografía",
    font=("Arial", 20, "bold")
).pack(pady=(15, 5))

ctk.CTkLabel(
    frame_img,
    text="Carga una imagen y aplica filtros por convolución.",
    font=("Arial", 13)
).pack(pady=(0, 10))

ctk.CTkButton(
    frame_img,
    text="Cargar imagen",
    command=cargar_imagen,
    height=38
).pack(padx=20, pady=6, fill="x")

filtro_var = ctk.StringVar(value="Sobel ambos")

filtros = [
    "Roberts",
    "Prewitt",
    "Sobel horizontal",
    "Sobel vertical",
    "Sobel ambos",
    "Laplace",
    "LOG"
]

ctk.CTkSegmentedButton(
    frame_img,
    values=filtros,
    variable=filtro_var
).pack(padx=20, pady=8, fill="x")

ctk.CTkButton(
    frame_img,
    text="Aplicar filtro seleccionado",
    command=aplicar_filtro,
    height=38
).pack(padx=20, pady=(8, 18), fill="x")


# ==========================
# SECCIÓN AUDIO
# ==========================

frame_audio = ctk.CTkFrame(contenedor, corner_radius=16)
frame_audio.pack(padx=20, pady=15, fill="x")

ctk.CTkLabel(
    frame_audio,
    text="Audio",
    font=("Arial", 20, "bold")
).pack(pady=(15, 5))

ctk.CTkLabel(
    frame_audio,
    text="Carga dos audios WAV. Se muestran sus señales, sus FFT hasta 4000 Hz y la convolución resultante.",
    font=("Arial", 13)
).pack(pady=(0, 10))

audio_grid = ctk.CTkFrame(frame_audio, fg_color="transparent")
audio_grid.pack(padx=20, pady=10, fill="x")

ctk.CTkButton(audio_grid, text="Cargar audio 1", command=cargar_audio_1).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
ctk.CTkButton(audio_grid, text="Cargar audio 2", command=cargar_audio_2).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
ctk.CTkButton(audio_grid, text="Convolucionar audios", command=convolucionar_audios).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

ctk.CTkButton(audio_grid, text="Reproducir audio 1", command=reproducir_audio_1).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
ctk.CTkButton(audio_grid, text="Reproducir audio 2", command=reproducir_audio_2).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
ctk.CTkButton(audio_grid, text="Reproducir convolucionado", command=reproducir_audio_convolucionado).grid(row=1, column=2, padx=5, pady=5, sticky="ew")

ctk.CTkButton(audio_grid, text="Señal audio 1", command=mostrar_senal_audio_1).grid(row=2, column=0, padx=5, pady=5, sticky="ew")
ctk.CTkButton(audio_grid, text="Señal audio 2", command=mostrar_senal_audio_2).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
ctk.CTkButton(audio_grid, text="Señal convolucionada", command=mostrar_senal_convolucionada).grid(row=2, column=2, padx=5, pady=5, sticky="ew")

ctk.CTkButton(audio_grid, text="FFT audio 1", command=mostrar_fft_audio_1).grid(row=3, column=0, padx=5, pady=5, sticky="ew")
ctk.CTkButton(audio_grid, text="FFT audio 2", command=mostrar_fft_audio_2).grid(row=3, column=1, padx=5, pady=5, sticky="ew")
ctk.CTkButton(audio_grid, text="FFT convolucionada", command=mostrar_fft_convolucionada).grid(row=3, column=2, padx=5, pady=5, sticky="ew")

ctk.CTkButton(
    audio_grid,
    text="Detener reproducción",
    command=detener_audio,
    fg_color="#b91c1c",
    hover_color="#7f1d1d"
).grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

audio_grid.grid_columnconfigure(0, weight=1)
audio_grid.grid_columnconfigure(1, weight=1)
audio_grid.grid_columnconfigure(2, weight=1)


# ==========================
# SECCIÓN VIDEO
# ==========================

frame_video = ctk.CTkFrame(contenedor, corner_radius=16)
frame_video.pack(padx=20, pady=15, fill="x")

ctk.CTkLabel(
    frame_video,
    text="Video",
    font=("Arial", 20, "bold")
).pack(pady=(15, 5))

ctk.CTkLabel(
    frame_video,
    text="Carga un video y aplica detección de bordes Sobel cuadro por cuadro.",
    font=("Arial", 13)
).pack(pady=(0, 10))

video_grid = ctk.CTkFrame(frame_video, fg_color="transparent")
video_grid.pack(padx=20, pady=8, fill="x")

ctk.CTkButton(
    video_grid,
    text="Cargar video",
    command=cargar_video
).grid(row=0, column=0, padx=5, pady=5, sticky="ew")

ctk.CTkButton(
    video_grid,
    text="Procesar video con Sobel",
    command=procesar_video
).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

video_grid.grid_columnconfigure(0, weight=1)
video_grid.grid_columnconfigure(1, weight=1)


# ==========================
# SALIR
# ==========================

ctk.CTkButton(
    ventana,
    text="Salir",
    command=ventana.destroy,
    fg_color="#b91c1c",
    hover_color="#7f1d1d",
    height=42
).pack(padx=25, pady=18, fill="x")

ventana.mainloop()