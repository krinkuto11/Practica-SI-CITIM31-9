import cv2
import numpy as np
import os
import shutil

#Verifica si el contorno dado tiene forma de rectángulo.
def es_rectangulo(contorno):

    epsilon = 0.03 * cv2.arcLength(contorno, True)  # Factor de aproximación (2% del perímetro)
    approx = cv2.approxPolyDP(contorno, epsilon, True)

    if len(approx) == 4:
        if cv2.isContourConvex(approx):
            return True

    return False

#Identifica el rectángulo más grande y transforma la imagen según los puntos del rectángulo
def encontrar_rectangulo(img_original):
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 10, 60, apertureSize=3)

    overlay = img_original.copy()
    overlay[edges != 0] = [0, 255, 0]

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_rect = None
    max_area = 0

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_rect = approx

    if largest_rect is not None:
        puntos = largest_rect.reshape(-1, 2)
        img_transformada = transformar_imagen(img_original, puntos)
    else:
        print("No se encontró rectangulo")

    return img_transformada

#devuelve el contorno mas grande despues de aplicar filtros
def encontrar_lineas(image_path):
    img_original = cv2.imread(image_path)

    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 20, 70, apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour, edges

#Dibuja las líneas que forman el contorno en la imagen original.
def dibujar_lineas_contorno(contorno):

    epsilon = 0.03 * cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, epsilon, True)

    puntos = approx.reshape(-1, 2)

    return puntos

#Transforma la imagen para que las líneas detectadas estén en el marco de la imagen.
def transformar_imagen(img_original, puntos):
    puntos = sorted(puntos, key=lambda x: (x[1], x[0]))
    top_points = sorted(puntos[:2], key=lambda x: x[0])
    bottom_points = sorted(puntos[2:], key=lambda x: x[0])
    ordered_points = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype="float32")

    width_a = np.linalg.norm(ordered_points[2] - ordered_points[3])
    width_b = np.linalg.norm(ordered_points[1] - ordered_points[0])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(ordered_points[1] - ordered_points[2])
    height_b = np.linalg.norm(ordered_points[0] - ordered_points[3])
    max_height = int(max(height_a, height_b))

    # Añadir margen
    margin_x = int(0.03 * max_width)
    margin_y = int(0.03 * max_height)
    max_width += 2 * margin_x
    max_height += 2 * margin_y

    destino = np.array([
        [margin_x, margin_y],
        [max_width - margin_x, margin_y],
        [max_width - margin_x, max_height - margin_y],
        [margin_x, max_height - margin_y]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_points, destino)
    img_transformada = cv2.warpPerspective(img_original, M, (max_width, max_height))

    return img_transformada


#detectar bordes, aplica filtro para cerrar huecos y encontrar el mayor contornos
def encontrar_lineas_nueva(image_path):
    img_original = cv2.imread(image_path)

    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 20, 70, apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour, edges


#Detecta contornos, segmenta y guarda cada contorno en una imagen.
def guardar_contornos_segmentados(img_transformada, folder_path):

    gray = cv2.cvtColor(img_transformada, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    os.makedirs(folder_path, exist_ok=True)

    img_with_rects = img_transformada.copy()

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Asegurar que el lado largo sea vertical
        if w > h:
            w, h = h, w

        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_transformada.shape[1] - x, w + 2 * margin)
        h = min(img_transformada.shape[0] - y, h + 2 * margin)

        cv2.rectangle(img_with_rects, (x, y), (x + w, y + h), (0, 255, 0), 2)

        segment = img_transformada[y:y+h, x:x+w]

        segment_path = os.path.join(folder_path, f"segment_{i+1}.png")
        cv2.imwrite(segment_path, segment)

    cv2.imshow("Rectangulos de Segmentacion", img_with_rects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Elimina el contenido de las carpetas IMG 1, IMG 2,... y las vuelve a crear vacías
def clear_segmented_folders(number_of_images, base_path="Resources/Segmentación/Output"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in range(1, number_of_images + 1):
        folder_path = os.path.join(base_path, f"IMG {i}")
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

#detecta los contornos de los caracteres y guarda cada uno dentro de 'output_folder'
def detectar_y_segmentar_caracteres(img, output_folder, margen=2):
    """
    .
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contour_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    img_con_contornos = img.copy()

    char_index = 1
    for (x, y, w, h) in contour_boxes:
        if w < 5 or h < 5:
            continue

        # Aplicar margen
        x1 = max(x - margen, 0)
        y1 = max(y - margen, 0)
        x2 = min(x + w + margen, img.shape[1])
        y2 = min(y + h + margen, img.shape[0])

        cv2.rectangle(img_con_contornos, (x1, y1), (x2, y2), (0, 255, 0), 2)  # <-- Se dibuja en verde

        char_img = img[y1:y2, x1:x2]

        char_path = os.path.join(output_folder, f"char_{char_index}.png")
        cv2.imwrite(char_path, char_img)

        char_index += 1

    cv2.imshow("Contornos detectados", img_con_contornos)  # <-- Ventana de visualización
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ordenar_regiones(regiones, tolerancia_altura=10):
    # Primero, ordenar por coordenada vertical (start_r)
    regiones_ordenadas = sorted(regiones, key=lambda r: r[0])
    regiones_agrupadas = []
    grupo_actual = []

    for reg in regiones_ordenadas:
        if not grupo_actual:
            grupo_actual.append(reg)
        else:
            # Comparar la altura de la región actual con la última región del grupo
            if abs(reg[0] - grupo_actual[-1][0]) < tolerancia_altura:
                grupo_actual.append(reg)
            else:
                # Ordenar el grupo actual por coordenada horizontal (sub_start) y agregar al resultado final
                grupo_actual = sorted(grupo_actual, key=lambda r: r[2])
                regiones_agrupadas.extend(grupo_actual)
                grupo_actual = [reg]
    # Procesar el último grupo
    if grupo_actual:
        grupo_actual = sorted(grupo_actual, key=lambda r: r[2])
        regiones_agrupadas.extend(grupo_actual)
    return regiones_agrupadas


if __name__ == "__main__":

    number_of_images = 7
    clear_segmented_folders(number_of_images, base_path="Resources/Segmentación")
    base_output_path = "Resources/Segmentación/Output"

    image_paths = [f"Resources/Segmentación/IMG {i}.jpeg" for i in range(1, 8)]

    for i, path in enumerate(image_paths, start=1):
        largest_contour, edges = encontrar_lineas_nueva(path)

        img_original = cv2.imread(path)

        img_largest_contour = img_original.copy()
        if len(largest_contour) > 0:
            cv2.drawContours(img_largest_contour, [largest_contour], -1, (0, 255, 0), 2)

        if es_rectangulo(largest_contour):
            puntos = dibujar_lineas_contorno(largest_contour)
            img_transformada = transformar_imagen(img_original, puntos)
        else:
            img_transformada = encontrar_rectangulo(img_original)

        img_gray = cv2.cvtColor(img_transformada, cv2.COLOR_BGR2GRAY)

        img_binaria = cv2.adaptiveThreshold(
            img_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # blockSize
            5  # C
        )

        cv2.imshow("Imagen original", img_original)
        # Muestra la imagen binaria
        cv2.imshow("Imagen transformada binaria", img_binaria)

        binaria_invertida = cv2.bitwise_not(img_binaria)

        projection = np.sum(binaria_invertida, axis=1)
        threshold_lineas = np.percentile(projection, 40)
        line_mask = projection > threshold_lineas

        lineas = []
        in_line = False
        line_start = 0
        current_mask = None

        for row_idx, active in enumerate(line_mask):
            if active:
                # Proyección horizontal para la fila actual
                current_projection = binaria_invertida[row_idx:row_idx + 1, :].sum(axis=0)
                current_active_columns = current_projection > (0.2 * np.max(current_projection))  # 20% del máximo valor

                if not in_line:
                    # Inicia un nuevo bloque
                    in_line = True
                    line_start = row_idx
                    current_mask = current_active_columns
                else:
                    # Verificar coincidencia con el bloque actual
                    overlap = np.logical_and(current_mask, current_active_columns)
                    if np.sum(overlap) < 0.8 * np.sum(current_active_columns):  # Menos del 50% de coincidencia
                        # No coincide, termina el bloque actual
                        line_end = row_idx
                        lineas.append((line_start, line_end))
                        line_start = row_idx
                        current_mask = current_active_columns
                    else:
                        # Actualizar la máscara del bloque actual
                        current_mask = np.logical_or(current_mask, current_active_columns)
            elif in_line:
                # Termina un bloque
                in_line = False
                line_end = row_idx
                lineas.append((line_start, line_end))

        if in_line:
            # Agregar el último bloque si está abierto
            lineas.append((line_start, len(line_mask)))

        h, w = binaria_invertida.shape[:2]

        umbral_altura_minima = h*0.04  # Ajusta este valor según las dimensiones de los caracteres esperados

        # Filtrar líneas basadas en su altura
        lineas_filtradas = [(start_r, end_r) for (start_r, end_r) in lineas if (end_r - start_r) > umbral_altura_minima]

        # Crear una copia de la imagen binaria invertida para visualizar las líneas filtradas
        img_lineas_filtradas = cv2.cvtColor(binaria_invertida, cv2.COLOR_GRAY2BGR)

        for (start_r, end_r) in lineas_filtradas:
            # Dibujar un rectángulo que cubra toda la línea
            cv2.rectangle(img_lineas_filtradas, (0, start_r), (img_lineas_filtradas.shape[1], end_r), (0, 255, 0), 2)

        # Mostrar las líneas seleccionadas después del filtrado
        cv2.imshow("Lineas filtradas", img_lineas_filtradas)

        umbral_vacío = 0.2  # Por debajo de este valor, una columna es "vacía"
        umbral_anchura_vacio = 0.01 * w  # Por encima de este valor, un vacío se considera suficientemente ancho
        umbral_anchura_minima = 0.05 * w
        umbral_concentracion_pixeles = 0.2

        # Crear una copia para visualizar las líneas y sus subregiones
        img_regiones = cv2.cvtColor(binaria_invertida, cv2.COLOR_GRAY2BGR)

        # Almacenar las subregiones finales (por línea y sublínea)
        regiones_finales = []

        for (start_r, end_r) in lineas_filtradas:
            # Recortar la línea actual
            line_img = binaria_invertida[start_r:end_r, :]

            # Proyección vertical (suma por columnas)
            col_projection = np.sum(line_img, axis=0)

            altura_linea = end_r - start_r
            umbral_columna_vacia = umbral_vacío * altura_linea

            # Detectar columnas vacías (por debajo del umbral)
            vacio_mask = col_projection < umbral_columna_vacia

            # Encontrar bloques consecutivos de columnas vacías
            vacios = []
            in_vacio = False
            vacio_start = 0
            for col_idx, is_vacio in enumerate(vacio_mask):
                if is_vacio and not in_vacio:
                    in_vacio = True
                    vacio_start = col_idx
                elif not is_vacio and in_vacio:
                    in_vacio = False
                    vacio_end = col_idx
                    vacios.append((vacio_start, vacio_end))
            if in_vacio:
                vacios.append((vacio_start, len(vacio_mask)))

            # Dividir las regiones llenas usando los vacíos anchos
            subregiones = []
            last_end = 0
            for (vacio_start, vacio_end) in vacios:
                ancho_vacio = vacio_end - vacio_start
                if ancho_vacio > umbral_anchura_vacio:
                    # Si el vacío es suficientemente ancho, dividir
                    subregiones.append((last_end, vacio_start))
                    last_end = vacio_end
            # Agregar la última subregión después del último vacío ancho
            subregiones.append((last_end, line_img.shape[1]))

            # Almacenar las subregiones con las coordenadas de la línea
            for (sub_start, sub_end) in subregiones:
                ancho_segmento = sub_end - sub_start
                altura_segmento = end_r - start_r

                # Filtrar segmentos muy finos o muy bajos
                if ancho_segmento >= umbral_anchura_minima and altura_segmento >= umbral_altura_minima:
                    segmento = line_img[:, sub_start:sub_end]

                    # Calcular la concentración de píxeles blancos en el segmento
                    pixeles_blancos = np.sum(segmento == 255)  # Píxeles blancos en la región
                    pixeles_totales = segmento.shape[0] * segmento.shape[1]  # Área total
                    concentracion_pixeles = pixeles_blancos / pixeles_totales

                    # Filtrar segmentos con baja concentración de píxeles blancos
                    if concentracion_pixeles >= umbral_concentracion_pixeles and ancho_segmento >= 1.3 * altura_segmento:
                        regiones_finales.append((start_r, end_r, sub_start, sub_end))

                        # Dibujar las subregiones válidas en la imagen para visualización
                        cv2.rectangle(
                            img_regiones,
                            (sub_start, start_r),
                            (sub_end, end_r),
                            (0, 255, 0),
                            2
                        )

        # Mostrar las regiones finales detectadas
        cv2.imshow("Regiones finales separadas", img_regiones)

        binaria_invertida = cv2.bitwise_not(binaria_invertida)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria_invertida, connectivity=4)

        # Filtrar componentes por tamaño (altura y anchura)
        umbral_altura_minima = h * 0.01  # Altura mínima aceptable para un carácter
        umbral_anchura_minima = w * 0.02  # Anchura mínima aceptable para un carácter
        umbral_altura_maxima = h * 0.2  # Altura maxima aceptable para un carácter
        umbral_anchura_maxima = w * 0.2  # Anchura maxima aceptable para un carácter

        img_resultado = cv2.cvtColor(binaria_invertida, cv2.COLOR_GRAY2BGR)

        # Dibujar las regiones anteriores en azul
        for (start_r, end_r, sub_start, sub_end) in regiones_finales:
            cv2.rectangle(
                img_resultado,
                (sub_start, start_r),
                (sub_end, end_r),
                (255, 0, 0),  # Azul
                2
            )

        mask_regiones_finales = np.zeros_like(binaria_invertida, dtype=np.uint8)

        for (start_r, end_r, sub_start, sub_end) in regiones_finales:
            cv2.rectangle(mask_regiones_finales, (sub_start, start_r), (sub_end, end_r), 255, thickness=-1)

        # Recorrer cada componente detectada y verificar si al menos un tercio está dentro de una región final
        for label in range(1, num_labels):  # El label 0 es el fondo
            x, y, w, h, area = stats[label]

            # Crear una máscara para el componente actual
            componente_mask = np.zeros_like(binaria_invertida, dtype=np.uint8)
            componente_mask[labels == label] = 255

            # Verificar intersección entre el componente actual y la máscara de regiones finales
            interseccion = cv2.bitwise_and(componente_mask, mask_regiones_finales)
            interseccion_area = np.sum(interseccion == 255)

            # Verificar si al menos un tercio del área del componente está dentro de una región final
            if interseccion_area >= area / 3:
                # Filtrar componentes demasiado pequeñas o demasiado grandes
                if (umbral_anchura_minima <= w <= umbral_anchura_maxima) and (
                        umbral_altura_minima <= h <= umbral_altura_maxima):
                    # Dibujar un rectángulo alrededor del carácter en verde
                    cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde

        # Mostrar las regiones detectadas
        cv2.imshow("Regiones anteriores (azul) y caracteres (verde)", img_resultado)


        # Ordenar las regiones finales
        regiones_ordenadas = ordenar_regiones(regiones_finales)
        img_visualizacion = cv2.cvtColor(binaria_invertida, cv2.COLOR_GRAY2BGR)
        total_regiones = len(regiones_ordenadas)

        # Preparar la información de cada componente detectado
        componentes_info = {}
        for label in range(1, num_labels):  # label 0 es fondo
            x, y, w_comp, h_comp, area = stats[label]
            comp_mask = np.zeros_like(binaria_invertida, dtype=np.uint8)
            comp_mask[labels == label] = 255
            componentes_info[label] = {
                "bbox": (x, y, w_comp, h_comp),
                "area": area,
                "mask": comp_mask
            }

        # Configurar carpeta de salida para la imagen actual
        output_folder = os.path.join(base_output_path, f"IMG {i}")
        os.makedirs(output_folder, exist_ok=True)
        char_count = 1  # Contador para los nombres de archivos de caracteres

        # Recorrer cada región ordenada y guardar caracteres en orden
        for idx_reg, (start_r, end_r, sub_start, sub_end) in enumerate(regiones_ordenadas):
            # Crear máscara para la región actual
            region_mask = np.zeros_like(binaria_invertida, dtype=np.uint8)
            cv2.rectangle(region_mask, (sub_start, start_r), (sub_end, end_r), 255, thickness=-1)

            componentes_en_region = []

            # Verificar intersección para cada componente con la región actual
            for label, info in componentes_info.items():
                x, y, w_comp, h_comp = info["bbox"]
                area = info["area"]
                comp_mask = info["mask"]

                interseccion = cv2.bitwise_and(comp_mask, region_mask)
                interseccion_area = np.sum(interseccion == 255)

                if interseccion_area >= area / 3:
                    componentes_en_region.append((label, x, y, w_comp, h_comp))

            # Ordenar componentes de izquierda a derecha dentro de la región
            componentes_en_region.sort(key=lambda comp: comp[1])

            # Guardar cada carácter de la región en orden
            for idx_char, (label, x, y, w_comp, h_comp) in enumerate(componentes_en_region):
                # Extraer la imagen del carácter usando las coordenadas detectadas
                char_img = img_transformada[y:y + h_comp, x:x + w_comp]

                min_area = 35

                # Verificar si la imagen cumple con las dimensiones mínimas
                if char_img.shape[1] * char_img.shape[0] < min_area:
                    continue

                margen = 2

                # Calcular las coordenadas con margen, ajustando para no salir de los límites de la imagen
                x_m = max(x - margen, 0)
                y_m = max(y - margen, 0)
                x2 = min(x + w_comp + margen, img_transformada.shape[1])
                y2 = min(y + h_comp + margen, img_transformada.shape[0])

                # Extraer la imagen del carácter con margen
                char_img = img_transformada[y_m:y2, x_m:x2]

                # Guardar el carácter en la carpeta correspondiente
                char_path = os.path.join(output_folder, f"char_{char_count}.png")
                cv2.imwrite(char_path, char_img)
                char_count += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def ejecucion_externa():
    number_of_images = 7
    clear_segmented_folders(number_of_images, base_path="Resources/Segmentación")
    base_output_path = "Resources/Segmentación/Output"

    image_paths = [f"Resources/Segmentación/IMG {i}.jpeg" for i in range(1, 8)]

    for i, path in enumerate(image_paths, start=1):
        largest_contour, edges = encontrar_lineas_nueva(path)

        img_original = cv2.imread(path)

        img_largest_contour = img_original.copy()
        if len(largest_contour) > 0:
            cv2.drawContours(img_largest_contour, [largest_contour], -1, (0, 255, 0), 2)

        if es_rectangulo(largest_contour):
            puntos = dibujar_lineas_contorno(largest_contour)
            img_transformada = transformar_imagen(img_original, puntos)
        else:
            img_transformada = encontrar_rectangulo(img_original)

        img_gray = cv2.cvtColor(img_transformada, cv2.COLOR_BGR2GRAY)

        img_binaria = cv2.adaptiveThreshold(
            img_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # blockSize
            5  # C
        )

        cv2.imshow("Imagen original", img_original)
        # Muestra la imagen binaria
        cv2.imshow("Imagen transformada binaria", img_binaria)

        binaria_invertida = cv2.bitwise_not(img_binaria)

        projection = np.sum(binaria_invertida, axis=1)
        threshold_lineas = np.percentile(projection, 40)
        line_mask = projection > threshold_lineas

        lineas = []
        in_line = False
        line_start = 0
        current_mask = None

        for row_idx, active in enumerate(line_mask):
            if active:
                # Proyección horizontal para la fila actual
                current_projection = binaria_invertida[row_idx:row_idx + 1, :].sum(axis=0)
                current_active_columns = current_projection > (0.2 * np.max(current_projection))  # 20% del máximo valor

                if not in_line:
                    # Inicia un nuevo bloque
                    in_line = True
                    line_start = row_idx
                    current_mask = current_active_columns
                else:
                    # Verificar coincidencia con el bloque actual
                    overlap = np.logical_and(current_mask, current_active_columns)
                    if np.sum(overlap) < 0.8 * np.sum(current_active_columns):  # Menos del 50% de coincidencia
                        # No coincide, termina el bloque actual
                        line_end = row_idx
                        lineas.append((line_start, line_end))
                        line_start = row_idx
                        current_mask = current_active_columns
                    else:
                        # Actualizar la máscara del bloque actual
                        current_mask = np.logical_or(current_mask, current_active_columns)
            elif in_line:
                # Termina un bloque
                in_line = False
                line_end = row_idx
                lineas.append((line_start, line_end))

        if in_line:
            # Agregar el último bloque si está abierto
            lineas.append((line_start, len(line_mask)))

        h, w = binaria_invertida.shape[:2]

        umbral_altura_minima = h*0.04  # Ajusta este valor según las dimensiones de los caracteres esperados

        # Filtrar líneas basadas en su altura
        lineas_filtradas = [(start_r, end_r) for (start_r, end_r) in lineas if (end_r - start_r) > umbral_altura_minima]

        # Crear una copia de la imagen binaria invertida para visualizar las líneas filtradas
        img_lineas_filtradas = cv2.cvtColor(binaria_invertida, cv2.COLOR_GRAY2BGR)

        for (start_r, end_r) in lineas_filtradas:
            # Dibujar un rectángulo que cubra toda la línea
            cv2.rectangle(img_lineas_filtradas, (0, start_r), (img_lineas_filtradas.shape[1], end_r), (0, 255, 0), 2)

        # Mostrar las líneas seleccionadas después del filtrado
        cv2.imshow("Lineas filtradas", img_lineas_filtradas)

        umbral_vacío = 0.2  # Por debajo de este valor, una columna es "vacía"
        umbral_anchura_vacio = 0.01 * w  # Por encima de este valor, un vacío se considera suficientemente ancho
        umbral_anchura_minima = 0.05 * w
        umbral_concentracion_pixeles = 0.2

        # Crear una copia para visualizar las líneas y sus subregiones
        img_regiones = cv2.cvtColor(binaria_invertida, cv2.COLOR_GRAY2BGR)

        # Almacenar las subregiones finales (por línea y sublínea)
        regiones_finales = []

        for (start_r, end_r) in lineas_filtradas:
            # Recortar la línea actual
            line_img = binaria_invertida[start_r:end_r, :]

            # Proyección vertical (suma por columnas)
            col_projection = np.sum(line_img, axis=0)

            altura_linea = end_r - start_r
            umbral_columna_vacia = umbral_vacío * altura_linea

            # Detectar columnas vacías (por debajo del umbral)
            vacio_mask = col_projection < umbral_columna_vacia

            # Encontrar bloques consecutivos de columnas vacías
            vacios = []
            in_vacio = False
            vacio_start = 0
            for col_idx, is_vacio in enumerate(vacio_mask):
                if is_vacio and not in_vacio:
                    in_vacio = True
                    vacio_start = col_idx
                elif not is_vacio and in_vacio:
                    in_vacio = False
                    vacio_end = col_idx
                    vacios.append((vacio_start, vacio_end))
            if in_vacio:
                vacios.append((vacio_start, len(vacio_mask)))

            # Dividir las regiones llenas usando los vacíos anchos
            subregiones = []
            last_end = 0
            for (vacio_start, vacio_end) in vacios:
                ancho_vacio = vacio_end - vacio_start
                if ancho_vacio > umbral_anchura_vacio:
                    # Si el vacío es suficientemente ancho, dividir
                    subregiones.append((last_end, vacio_start))
                    last_end = vacio_end
            # Agregar la última subregión después del último vacío ancho
            subregiones.append((last_end, line_img.shape[1]))

            # Almacenar las subregiones con las coordenadas de la línea
            for (sub_start, sub_end) in subregiones:
                ancho_segmento = sub_end - sub_start
                altura_segmento = end_r - start_r

                # Filtrar segmentos muy finos o muy bajos
                if ancho_segmento >= umbral_anchura_minima and altura_segmento >= umbral_altura_minima:
                    segmento = line_img[:, sub_start:sub_end]

                    # Calcular la concentración de píxeles blancos en el segmento
                    pixeles_blancos = np.sum(segmento == 255)  # Píxeles blancos en la región
                    pixeles_totales = segmento.shape[0] * segmento.shape[1]  # Área total
                    concentracion_pixeles = pixeles_blancos / pixeles_totales

                    # Filtrar segmentos con baja concentración de píxeles blancos
                    if concentracion_pixeles >= umbral_concentracion_pixeles and ancho_segmento >= 1.3 * altura_segmento:
                        regiones_finales.append((start_r, end_r, sub_start, sub_end))

                        # Dibujar las subregiones válidas en la imagen para visualización
                        cv2.rectangle(
                            img_regiones,
                            (sub_start, start_r),
                            (sub_end, end_r),
                            (0, 255, 0),
                            2
                        )

        # Mostrar las regiones finales detectadas
        cv2.imshow("Regiones finales separadas", img_regiones)

        binaria_invertida = cv2.bitwise_not(binaria_invertida)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria_invertida, connectivity=4)

        # Filtrar componentes por tamaño (altura y anchura)
        umbral_altura_minima = h * 0.01  # Altura mínima aceptable para un carácter
        umbral_anchura_minima = w * 0.02  # Anchura mínima aceptable para un carácter
        umbral_altura_maxima = h * 0.2  # Altura maxima aceptable para un carácter
        umbral_anchura_maxima = w * 0.2  # Anchura maxima aceptable para un carácter

        img_resultado = cv2.cvtColor(binaria_invertida, cv2.COLOR_GRAY2BGR)

        # Dibujar las regiones anteriores en azul
        for (start_r, end_r, sub_start, sub_end) in regiones_finales:
            cv2.rectangle(
                img_resultado,
                (sub_start, start_r),
                (sub_end, end_r),
                (255, 0, 0),  # Azul
                2
            )

        mask_regiones_finales = np.zeros_like(binaria_invertida, dtype=np.uint8)

        for (start_r, end_r, sub_start, sub_end) in regiones_finales:
            cv2.rectangle(mask_regiones_finales, (sub_start, start_r), (sub_end, end_r), 255, thickness=-1)

        # Recorrer cada componente detectada y verificar si al menos un tercio está dentro de una región final
        for label in range(1, num_labels):  # El label 0 es el fondo
            x, y, w, h, area = stats[label]

            # Crear una máscara para el componente actual
            componente_mask = np.zeros_like(binaria_invertida, dtype=np.uint8)
            componente_mask[labels == label] = 255

            # Verificar intersección entre el componente actual y la máscara de regiones finales
            interseccion = cv2.bitwise_and(componente_mask, mask_regiones_finales)
            interseccion_area = np.sum(interseccion == 255)

            # Verificar si al menos un tercio del área del componente está dentro de una región final
            if interseccion_area >= area / 3:
                # Filtrar componentes demasiado pequeñas o demasiado grandes
                if (umbral_anchura_minima <= w <= umbral_anchura_maxima) and (
                        umbral_altura_minima <= h <= umbral_altura_maxima):
                    # Dibujar un rectángulo alrededor del carácter en verde
                    cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde

        # Mostrar las regiones detectadas
        cv2.imshow("Regiones anteriores (azul) y caracteres (verde)", img_resultado)


        # Ordenar las regiones finales
        regiones_ordenadas = ordenar_regiones(regiones_finales)
        img_visualizacion = cv2.cvtColor(binaria_invertida, cv2.COLOR_GRAY2BGR)
        total_regiones = len(regiones_ordenadas)

        # Preparar la información de cada componente detectado
        componentes_info = {}
        for label in range(1, num_labels):  # label 0 es fondo
            x, y, w_comp, h_comp, area = stats[label]
            comp_mask = np.zeros_like(binaria_invertida, dtype=np.uint8)
            comp_mask[labels == label] = 255
            componentes_info[label] = {
                "bbox": (x, y, w_comp, h_comp),
                "area": area,
                "mask": comp_mask
            }

        # Configurar carpeta de salida para la imagen actual
        output_folder = os.path.join(base_output_path, f"IMG {i}")
        os.makedirs(output_folder, exist_ok=True)
        char_count = 1  # Contador para los nombres de archivos de caracteres

        # Recorrer cada región ordenada y guardar caracteres en orden
        for idx_reg, (start_r, end_r, sub_start, sub_end) in enumerate(regiones_ordenadas):
            # Crear máscara para la región actual
            region_mask = np.zeros_like(binaria_invertida, dtype=np.uint8)
            cv2.rectangle(region_mask, (sub_start, start_r), (sub_end, end_r), 255, thickness=-1)

            componentes_en_region = []

            # Verificar intersección para cada componente con la región actual
            for label, info in componentes_info.items():
                x, y, w_comp, h_comp = info["bbox"]
                area = info["area"]
                comp_mask = info["mask"]

                interseccion = cv2.bitwise_and(comp_mask, region_mask)
                interseccion_area = np.sum(interseccion == 255)

                if interseccion_area >= area / 3:
                    componentes_en_region.append((label, x, y, w_comp, h_comp))

            # Ordenar componentes de izquierda a derecha dentro de la región
            componentes_en_region.sort(key=lambda comp: comp[1])

            # Guardar cada carácter de la región en orden
            for idx_char, (label, x, y, w_comp, h_comp) in enumerate(componentes_en_region):
                # Extraer la imagen del carácter usando las coordenadas detectadas
                char_img = img_transformada[y:y + h_comp, x:x + w_comp]

                min_area = 35

                # Verificar si la imagen cumple con las dimensiones mínimas
                if char_img.shape[1] * char_img.shape[0] < min_area:
                    continue

                margen = 2

                # Calcular las coordenadas con margen, ajustando para no salir de los límites de la imagen
                x_m = max(x - margen, 0)
                y_m = max(y - margen, 0)
                x2 = min(x + w_comp + margen, img_transformada.shape[1])
                y2 = min(y + h_comp + margen, img_transformada.shape[0])

                # Extraer la imagen del carácter con margen
                char_img = img_transformada[y_m:y2, x_m:x2]

                # Guardar el carácter en la carpeta correspondiente
                char_path = os.path.join(output_folder, f"char_{char_count}.png")
                cv2.imwrite(char_path, char_img)
                char_count += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()