# Modulos estandar
import numpy as np
import matplotlib.pyplot as plt
import time

# Modulos para el procesamiento de imagenes
import skimage.io as io
import skimage.exposure as exposure
import skimage.measure as measure
import skimage.color as color
import skimage.morphology as morph
import scipy.signal as sig

def mostrar_imagen(imagen):
    
    # Obtain image shape and print relevant info
    pixels_y, pixels_x, depth = imagen.shape
    print("Su imagen tiene las siguientes dimensiones:\nPixeles en Y: {}\nPixeles en X: {}\nCanales de color: {}".format(pixels_y, pixels_x, depth))
    
    plt.figure(figsize=(5,5), dpi=200)
    plt.title("Imagen original")
    plt.ylabel("Pixeles en Y")
    plt.xlabel("Pixeles en X")
    plt.imshow(imagen)
    
    return None

def normalizar_img(imagen):
    
    # Nomalize the range of values in a given image
    norm_img = (imagen - imagen.min())/(imagen.max() - imagen.min())
    
    return norm_img

def mostrar_histograma_RGB(imagen):
    
    # Compute histograms of R, G and B channels individually
    hist_r, bins_r = exposure.histogram(imagen[:,:,0])
    hist_g, bins_g = exposure.histogram(imagen[:,:,1])
    hist_b, bins_b = exposure.histogram(imagen[:,:,2])

    fig, axs = plt.subplots(1, 3, figsize=(9,3), dpi=200)

    # Red channel
    axs[0].plot(bins_r, hist_r)
    axs[0].set_title("Histograma del Canal R")
    axs[0].set_ylabel("Numero de pixeles")
    axs[0].set_xlabel("Valor del Pixel (U.A)")
    # Green channel
    axs[1].plot(bins_g, hist_g)
    axs[1].set_title("Histograma del Canal G")
    axs[1].set_ylabel("Numero de pixeles")
    axs[1].set_xlabel("Valor del Pixel (U.A)")
    # Blue channel
    axs[2].plot(bins_b, hist_b)
    axs[2].set_title("Histograma del Canal B")
    axs[2].set_ylabel("Numero de pixeles")
    axs[2].set_xlabel("Valor del Pixel (U.A)")
    
    plt.tight_layout()
    
    return None

def obtener_seccion(imagen):
    
    # Obtain image shape and print relevant info
    pixels_y, pixels_x, depth = imagen.shape
    print("Su imagen tiene las siguientes dimensiones:\nPixeles en Y: {}\nPixeles en X: {}\nCanales de color: {}".format(pixels_y, pixels_x, depth))
    
    # Obtain a section of an image
    xi = int(input("Ingrese pixel de inicio en X: "))
    xf = int(input("Ingrese pixel final en X: "))
    yi = int(input("Ingrese pixel de inicio en Y: "))
    yf = int(input("Ingrese pixel final en Y: "))
    
    # Define a copy of the image
    cut_img = imagen[yi:yf,xi:xf,:]
    
    return cut_img

def mostrar_canales_RGB(imagen, fila):
    
    # Generate a figure and axes
    fig, axs = plt.subplots(3, 3, figsize=(9,9), dpi=200, sharex="col")

    # Red channel
    axs[0][0].imshow(normalizar_img(imagen[:,:,0]), cmap="Reds")
    axs[0][0].set_title("Canal R")
    axs[0][0].set_ylabel("Pixel en Y")
    axs[0][0].set_xlabel("Pixel en X")
    fig.colorbar(axs[0][0].imshow(normalizar_img(imagen[:,:,0]), cmap="Reds"), ax=axs[0][0])

    # Plot line and histogram for line
    axs[0][0].hlines(fila, 0, imagen.shape[1], colors="k", linestyle="dashed", lw=1.0, alpha=0.5)
    img_norm_r = normalizar_img(imagen[fila,:,0])
    hist_r, bins_r = exposure.histogram(img_norm_r)
    axs[0][2].plot(bins_r, hist_r, c="r", lw=1.0, alpha=0.5)
    axs[0][2].set_title("Histograma Canal R")
    axs[0][2].set_ylabel("Número de pixeles")
    axs[0][2].set_xlabel("Valor del pixel")
    
    # Plot individual pixel values
    x_r = [i for i in range(0,imagen.shape[1])]
    axs[0][1].plot(x_r, normalizar_img(imagen[fila,:,0]), c="r", alpha=0.8)
    axs[0][1].set_title("Valores de pixeles del Canal R")
    axs[0][1].set_ylabel("Valor del pixel")
    axs[0][1].set_xlabel("Pixel en X")
    
    # Change axes text size
    plt.setp(axs[0][0].get_xticklabels(), fontsize=8)
    plt.setp(axs[0][0].get_yticklabels(), fontsize=8)
    plt.setp(axs[0][1].get_xticklabels(), fontsize=8)
    plt.setp(axs[0][1].get_yticklabels(), fontsize=8)
    plt.setp(axs[0][2].get_xticklabels(), fontsize=8)
    plt.setp(axs[0][2].get_yticklabels(), fontsize=8)

    # Green channel
    axs[1][0].imshow(normalizar_img(imagen[:,:,1]), cmap="Greens")
    axs[1][0].set_title("Canal G")
    axs[1][0].set_ylabel("Pixel en Y")
    axs[1][0].set_xlabel("Pixel en X")
    fig.colorbar(axs[1][0].imshow(normalizar_img(imagen[:,:,1]), cmap="Greens"), ax=axs[1][0])

    # Plot line and histogram for line
    axs[1][0].hlines(fila, 0, imagen.shape[1], colors="k", linestyle="dashed", lw=1.0, alpha=0.3)
    img_norm_g = normalizar_img(imagen[fila,:,1])
    hist_g, bins_g = exposure.histogram(img_norm_g)
    axs[1][2].plot(bins_g, hist_g, c="g", lw=1.0, alpha=0.7)
    axs[1][2].set_title("Histograma Canal G")
    axs[1][2].set_ylabel("Número de pixeles")
    axs[1][2].set_xlabel("Valor del pixel")
    
    # Plot individual pixel values
    x_g = [i for i in range(0,imagen.shape[1])]
    axs[1][1].plot(x_g, normalizar_img(imagen[fila,:,1]), c="g", alpha=0.8)
    axs[1][1].set_title("Valores de pixeles del Canal R")
    axs[1][1].set_ylabel("Valor del pixel")
    axs[1][1].set_xlabel("Pixel en X")
    
    # Change axes text size
    plt.setp(axs[1][0].get_xticklabels(), fontsize=8)
    plt.setp(axs[1][0].get_yticklabels(), fontsize=8)
    plt.setp(axs[1][1].get_xticklabels(), fontsize=8)
    plt.setp(axs[1][1].get_yticklabels(), fontsize=8)
    plt.setp(axs[1][2].get_xticklabels(), fontsize=8)
    plt.setp(axs[1][2].get_yticklabels(), fontsize=8)

    # Blue channel
    axs[2][0].imshow(normalizar_img(imagen[:,:,2]), cmap="Blues")
    axs[2][0].set_title("Canal B")
    axs[2][0].set_ylabel("Pixel en Y")
    axs[2][0].set_xlabel("Pixel en X")
    fig.colorbar(axs[2][0].imshow(normalizar_img(imagen[:,:,2]), cmap="Blues"), ax=axs[2][0])

    # Plot line and histogram for line
    axs[2][0].hlines(fila, 0, imagen.shape[1], colors="k", linestyle="dashed", lw=1.0, alpha=0.5)
    img_norm_b = normalizar_img(imagen[fila,:,2])
    hist_b, bins_b = exposure.histogram(img_norm_b)
    axs[2][2].plot(bins_b, hist_b, c="b", lw=1.0, alpha=0.7)
    axs[2][2].set_title("Histgrama Canal B")
    axs[2][2].set_ylabel("Número de pixeles")
    axs[2][2].set_xlabel("Valor del pixel")
    
    # Plot individual pixel values
    x_b = [i for i in range(0,imagen.shape[1])]
    axs[2][1].plot(x_b, normalizar_img(imagen[fila,:,2]), c="b", alpha=0.8)
    axs[2][1].set_title("Valores de pixeles del Canal R")
    axs[2][1].set_ylabel("Valor del pixel")
    axs[2][1].set_xlabel("Pixel en X")
    
    # Change axes text size
    plt.setp(axs[2][0].get_xticklabels(), fontsize=8)
    plt.setp(axs[2][0].get_yticklabels(), fontsize=8)
    plt.setp(axs[2][1].get_xticklabels(), fontsize=8)
    plt.setp(axs[2][1].get_yticklabels(), fontsize=8)
    plt.setp(axs[2][2].get_xticklabels(), fontsize=8)
    plt.setp(axs[2][2].get_yticklabels(), fontsize=8)

    plt.tight_layout()
    
    return None

def mostrar_canal_greyscale(imagen, fila):
    
    # Image is converted to greyscale (if not already greyscale)
    try:
        if imagen.shape[2] == 3:
            img_grey = color.rgb2gray(imagen)
    except:
        img_grey = imagen
        
    # Plotting imagen and histograms
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=200)
    
    # Greyscale channel
    axs[0].imshow(normalizar_img(img_grey), cmap="Reds")
    axs[0].set_title("Escala de grises")
    axs[0].set_ylabel("Pixel en Y")
    axs[0].set_xlabel("Pixel en X")
    fig.colorbar(axs[0].imshow(normalizar_img(img_grey), cmap="Greys"), ax=axs[0])

    # Plot line and histogram for line
    axs[0].hlines(fila, 0, img_grey.shape[1], colors="k", linestyle="dashed", lw=1.0, alpha=0.5)
    img_norm = normalizar_img(img_grey[fila,:])
    hist, bins = exposure.histogram(img_norm)
    axs[2].plot(bins, hist, c="k", lw=1.0, alpha=0.7)
    axs[2].set_title("Histograma de escala de grises")
    axs[2].set_ylabel("Número de pixeles")
    axs[2].set_xlabel("Valor del pixel")
    
    # Plot individual pixel values
    x_v = [i for i in range(0,img_grey.shape[1])]
    axs[1].plot(x_v, normalizar_img(img_grey[fila,:]), c="k", alpha=0.8)
    axs[1].set_title("Valores de pixeles en escala de grises")
    axs[1].set_ylabel("Valor del pixel")
    axs[1].set_xlabel("Pixel en X")
    
    # Change axes text size

    plt.setp(axs[0].get_xticklabels(), fontsize=8)
    plt.setp(axs[0].get_yticklabels(), fontsize=8)
    plt.setp(axs[1].get_xticklabels(), fontsize=8)
    plt.setp(axs[1].get_yticklabels(), fontsize=8)
    plt.setp(axs[2].get_xticklabels(), fontsize=8)
    plt.setp(axs[2].get_yticklabels(), fontsize=8)
    
    plt.tight_layout()
    
    # Find local maxima for each tube
    peaks = sig.find_peaks(x_v)
    #for peak, i in zip(peaks, list(range(peaks))):
    #    valor = x_v[peak]
    #    print("Maximo tubo {} = {}\n".format(i+1, valor))
    
    print(peaks)
    return None

def convertir_escala_de_grises(imagen):
    
    return color.rgb2gray(imagen)

def mostrar_canales_HSV(imagen, fila):
    
    img_hsv = color.rgb2hsv(imagen)
    
    # Generate a figure and axes
    fig, axs = plt.subplots(3, 3, figsize=(9,9), dpi=200, sharex="col")

    # Red channel
    axs[0][0].imshow(normalizar_img(img_hsv[:,:,0]), cmap="hsv")
    axs[0][0].set_title("Canal H")
    axs[0][0].set_ylabel("Pixel en Y")
    axs[0][0].set_xlabel("Pixel en X")
    fig.colorbar(axs[0][0].imshow(normalizar_img(img_hsv[:,:,0]), cmap="hsv"), ax=axs[0][0])

    # Plot line and histogram for line
    axs[0][0].hlines(fila, 0, img_hsv.shape[1], colors="k", linestyle="dashed", lw=1.0, alpha=0.5)
    img_norm_r = normalizar_img(img_hsv[fila,:,0])
    hist_r, bins_r = exposure.histogram(img_norm_r)
    axs[0][2].plot(bins_r, hist_r, c="k", lw=1.0, alpha=0.5)
    axs[0][2].set_title("Histograma Canal H")
    axs[0][2].set_ylabel("Número de pixeles")
    axs[0][2].set_xlabel("Valor del pixel")
    
    # Plot individual pixel values
    x_h = [i for i in range(0,img_hsv.shape[1])]
    axs[0][1].plot(x_h, normalizar_img(img_hsv[fila,:,1]), c="k", alpha=0.8)
    axs[0][1].set_title("Valores de pixeles del Canal H")
    axs[0][1].set_ylabel("Valor del pixel")
    axs[0][1].set_xlabel("Pixel en X")
    
    # Change axes text size
    plt.setp(axs[0][0].get_xticklabels(), fontsize=8)
    plt.setp(axs[0][0].get_yticklabels(), fontsize=8)
    plt.setp(axs[0][1].get_xticklabels(), fontsize=8)
    plt.setp(axs[0][1].get_yticklabels(), fontsize=8)
    plt.setp(axs[0][2].get_xticklabels(), fontsize=8)
    plt.setp(axs[0][2].get_yticklabels(), fontsize=8)

    # Green channel
    axs[1][0].imshow(normalizar_img(img_hsv[:,:,1]), cmap="hsv")
    axs[1][0].set_title("Canal S")
    axs[1][0].set_ylabel("Pixel en Y")
    axs[1][0].set_xlabel("Pixel en X")
    fig.colorbar(axs[1][0].imshow(normalizar_img(img_hsv[:,:,1]), cmap="hsv"), ax=axs[1][0])

    # Plot line and histogram for line
    axs[1][0].hlines(fila, 0, img_hsv.shape[1], colors="k", linestyle="dashed", lw=1.0, alpha=0.3)
    img_norm_g = normalizar_img(img_hsv[fila,:,1])
    hist_g, bins_g = exposure.histogram(img_norm_g)
    axs[1][2].plot(bins_g, hist_g, c="k", lw=1.0, alpha=0.7)
    axs[1][2].set_title("Histograma Canal S")
    axs[1][2].set_ylabel("Número de pixeles")
    axs[1][2].set_xlabel("Valor del pixel")
    
    # Plot individual pixel values
    x_s = [i for i in range(0,img_hsv.shape[1])]
    axs[1][1].plot(x_s, normalizar_img(img_hsv[fila,:,1]), c="k", alpha=0.8)
    axs[1][1].set_title("Valores de pixeles del Canal S")
    axs[1][1].set_ylabel("Valor del pixel")
    axs[1][1].set_xlabel("Pixel en X")
    
    # Change axes text size
    plt.setp(axs[1][0].get_xticklabels(), fontsize=8)
    plt.setp(axs[1][0].get_yticklabels(), fontsize=8)
    plt.setp(axs[1][1].get_xticklabels(), fontsize=8)
    plt.setp(axs[1][1].get_yticklabels(), fontsize=8)
    plt.setp(axs[1][2].get_xticklabels(), fontsize=8)
    plt.setp(axs[1][2].get_yticklabels(), fontsize=8)

    # Blue channel
    axs[2][0].imshow(normalizar_img(img_hsv[:,:,2]), cmap="hsv")
    axs[2][0].set_title("Canal V")
    axs[2][0].set_ylabel("Pixel en Y")
    axs[2][0].set_xlabel("Pixel en X")
    fig.colorbar(axs[2][0].imshow(normalizar_img(img_hsv[:,:,2]), cmap="hsv"), ax=axs[2][0])

    # Plot line and histogram for line
    axs[2][0].hlines(fila, 0, img_hsv.shape[1], colors="k", linestyle="dashed", lw=1.0, alpha=0.5)
    img_norm_b = normalizar_img(img_hsv[fila,:,2])
    hist_b, bins_b = exposure.histogram(img_norm_b)
    axs[2][2].plot(bins_b, hist_b, c="k", lw=1.0, alpha=0.7)
    axs[2][2].set_title("Histgrama Canal V")
    axs[2][2].set_ylabel("Número de pixeles")
    axs[2][2].set_xlabel("Valor del pixel")
    
    # Plot individual pixel values
    x_v = [i for i in range(0,img_hsv.shape[1])]
    axs[2][1].plot(x_v, normalizar_img(img_hsv[fila,:,2]), c="k", alpha=0.8)
    axs[2][1].set_title("Valores de pixeles del Canal V")
    axs[2][1].set_ylabel("Valor del pixel")
    axs[2][1].set_xlabel("Pixel en X")
    
    # Change axes text size
    plt.setp(axs[2][0].get_xticklabels(), fontsize=8)
    plt.setp(axs[2][0].get_yticklabels(), fontsize=8)
    plt.setp(axs[2][1].get_xticklabels(), fontsize=8)
    plt.setp(axs[2][1].get_yticklabels(), fontsize=8)
    plt.setp(axs[2][2].get_xticklabels(), fontsize=8)
    plt.setp(axs[2][2].get_yticklabels(), fontsize=8)

    plt.tight_layout()
    
    return None

def convertir_espacio_hsv(imagen):
    
    return normalizar_img(color.rgb2hsv(imagen))

def segmentar_imagen(imagen):
    
    # First ask the user for an image type, channel and threshold value to use
    image_type = str(input("Por favor ingrese el TIPO de imagen a utilizar\n(RGB, GRIS, HSV): "))
    if image_type.upper() not in ["RGB", "GRIS", "HSV"]:
        print("Por favor ingrese un tipo de imagen valido! (RGB, GRIS, HSV)")
        return None
    image_channel = int(input("Por favor ingrese el CANAL que utilizará para realizar la segmentación\n(ingrese su opcion como numero): "))
    image_threshold = float(input("Por favor ingrese el UMBRAL que utilizará para segmentar\n(entre 0 y 1)"))
    thresh_up_low = str(input("Indique si quiere segmentar aquello MAYOR que el umbral o MENOR que el umbral\n(ingrese su opcion como texto): "))
    if thresh_up_low.upper() not in ["MAYOR", "MENOR"]:
        print("Por favor ingrese MAYOR o MENOR!")
        return None
    
    if image_type.upper() == "GRIS":
        
        if thresh_up_low.upper() == "MAYOR":        
            thresholded_image = imagen>image_threshold
        elif thresh_up_low.upper() == "MENOR":
            thresholded_image = imagen<image_threshold
        
        return thresholded_image
    
    elif image_type.upper() == "RGB" or image_type.upper() == "HSV":
        
        if thresh_up_low.upper() == "MAYOR":
            thresholded_image = imagen[:,:,image_channel]>image_threshold
        elif thresh_up_low.upper() == "MENOR":
            thresholded_image = imagen[:,:,image_channel]<image_threshold
            
        return thresholded_image
    
def limpiar_imagen(imagen, tamaño_minimo):
    
    # For removal of small objects from the binary image
    clean_img = morph.remove_small_objects(imagen, min_size=tamaño_minimo)
    
    return clean_img

def obtener_info_tubos(imagen_segmentada, area_de_interes):
    
    # Obtain tubes, label them and calculate the intensity
    
    # Rotate array clockwise for correct labeling of tubes
    rotated_array = np.rot90(imagen_segmentada, axes=(1,0))
    
    # Label each object and asign a number to it
    labeled_obj, num_obj = measure.label(rotated_array, return_num=True)

    # Rotate array 90 degrees counter-clockwise to obtain labeled image
    labeled_obj = np.rot90(labeled_obj, axes=(0,1))
    
    # See the results
    print("El algoritmo ha detectado {} tubos postitivos, a continuacion se graficará la imagen".format(num_obj))
    time.sleep(2)
    
    plt.figure(figsize=(5,5), dpi=200)
    plt.title("Objetos detectados")
    plt.imshow(labeled_obj, cmap="viridis")
    plt.colorbar()
    plt.show()
    
    time.sleep(2)
    
    # Prompt the user for input; to continue or cancel and tweak the other algorithms
    continuar = int(input("Desea continuar con el analisis o desea volver y realizar ajustes a algoritmos previos?\n1 = Continuar\n0 = Cancelar y volver\n"))
    
    # If user continues, multiply segmented image by each channel of the original image and stack them on third dimension
    if continuar:
        
        # Generate dictionary for storage of data
        results_dict = {}
        
        # Go object by object to obtain the intensity
        for i in range(num_obj):
            
            tube = (labeled_obj == i+1)
            # Tube area
            tube_area = np.sum(tube)
            # Multiply by greyscale image
            greyscale = np.multiply(color.rgb2gray(area_de_interes), tube)
            # Obtain average intensity (sum all pixel values and divide by area)
            avg_intensity = np.divide(np.sum(greyscale), tube_area)
            
            results_dict["tubo_positivo_{}".format(i+1)] = avg_intensity
            print("A continuacion se le entregara la informacion de la intensidad media de cada tubo positivo:\n")
        
        return results_dict
        
    else:
        
        print("Se cancela la operación, puede volver y realizar los ajustes que considere pertinentes")
        return None
    
    return None

def cargar_imagen(nombre_archivo):
    
    imagen = io.imread(nombre_archivo)
    
    # Show user the image
    # Obtain image shape and print relevant info
    pixels_y, pixels_x, depth = imagen.shape
    print("Su imagen tiene las siguientes dimensiones:\nPixeles en Y: {}\nPixeles en X: {}\nCanales de color: {}\n\n\n".format(pixels_y, pixels_x, depth))
    
    time.sleep(2)
    
    plt.figure(figsize=(3,3), dpi=200)
    plt.title("Imagen original")
    plt.ylabel("Pixeles en Y")
    plt.xlabel("Pixeles en X")
    plt.imshow(imagen)
    plt.show()
    
    time.sleep(2)
    # Ask user if rotation is necessary
    rotate = int(input("\n\nNecesita rotar su imagen?\n1 = Si\n0 = No\n"))
    
    if rotate:
        finished = False
        while not finished:
            direction = str(input("\nPor favor indique hacia qué direccion desea rotar la imagen\nLEFT = Izquierda\nRIGHT = Derecha\n"))
            degrees = int(input("\nPor favor indique cuantas veces desea rotar la imagen en 90 grados\n1 = 90\n2 = 180\n3 = 270\n"))
            
            if direction.upper() == "LEFT" or direction.upper() == "IZQUIERDA":
                axs = (1,0)
                
            elif direction.upper() == "RIGHT" or direction.upper() == "DERECHA":
                axs = (0,1)                
            
            time.sleep(2)
            # Rotate image and show the user, ask for feedback
            img_rotada = np.rot90(imagen, k=degrees,axes=axs)
            plt.figure(figsize=(3,3), dpi=200)
            plt.imshow(img_rotada)
            plt.show()
            
            time.sleep(2)
            # Ask user if image is correctly rotated
            listo = str(input("\n\nDesea rotar nuevamente la imagen o finalizar el proceso de importacion?\n0 = Seguir intentando la rotacion\n1 = Finalizar el proceso de importacion\n"))
            
            finished = listo
        print("\n\nHa finalizado su importación")
        return img_rotada
    
    else:
        print("\n\nHa finalizado su importación")
        return imagen
        
    return None