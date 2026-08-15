import numpy as np 
import matplotlib.pyplot as plt

def dibujar_linea(gradiente, x_centro, y_centro, distancia, eje):
    """
    Dibuja una línea tangente que representa el gradiente en el punto (x_centro, y_centro).
    """
    x_vals = np.linspace(x_centro - distancia, x_centro + distancia, 50)
    y_vals = gradiente * (x_vals - x_centro) + y_centro

    eje.scatter(x_centro, y_centro, color='b', s=50)
    eje.plot(x_vals, y_vals, linestyle='--', color='r', zorder=10, linewidth=1)

    desplazamiento_x = 30 if x_centro == 200 else 10
    texto = r"$\frac{\partial J}{\partial w}$ = %d" % gradiente

    eje.annotate(texto,
                 fontsize=14,
                 xy=(x_centro, y_centro),
                 xycoords='data',
                 xytext=(desplazamiento_x, 10),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"),
                 ha='left',
                 va='top')

def graficar_gradientes(x_datos, y_etiquetas, funcion_costo, funcion_gradiente):
    """
    Visualiza la relación entre el costo y el parámetro w, y muestra el campo de gradientes
    en un gráfico de vectores.
    """
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4))

    # ============
    # Primer gráfico
    # ============
    b_fijo = 100
    valores_w = np.linspace(0, 400, 50)
    costos = np.array([funcion_costo(x_datos, y_etiquetas, w, b_fijo) for w in valores_w])

    ejes[0].plot(valores_w, costos, linewidth=1)
    ejes[0].set_title("Costo vs w con gradiente (b = 100)")
    ejes[0].set_xlabel("w")
    ejes[0].set_ylabel("Costo")

    # Añadir líneas de gradiente para ciertos valores de w
    for w_prueba in [50, 200, 250]:
        grad_w, _ = funcion_gradiente(x_datos, y_etiquetas, w_prueba, b_fijo)
        costo_actual = funcion_costo(x_datos, y_etiquetas, w_prueba, b_fijo)
        dibujar_linea(grad_w, w_prueba, costo_actual, 30, ejes[0])

    # ===============
    # Segundo gráfico
    # ===============
    b_grid, w_grid = np.meshgrid(np.linspace(-200, 200, 10),
                                 np.linspace(-100, 600, 10))
    grad_w_matriz = np.zeros_like(w_grid)
    grad_b_matriz = np.zeros_like(b_grid)

    for i in range(w_grid.shape[0]):
        for j in range(w_grid.shape[1]):
            grad_w_matriz[i, j], grad_b_matriz[i, j] = funcion_gradiente(
                x_datos, y_etiquetas, w_grid[i, j], b_grid[i, j]
            )

    magnitud_color = np.sqrt((grad_b_matriz / 2)**2 + (grad_w_matriz / 2)**2)

    ejes[1].set_title("Campo de gradientes (gráfico vectorial)")
    Q = ejes[1].quiver(w_grid, b_grid, grad_w_matriz, grad_b_matriz,
                       magnitud_color, units='width')
    ejes[1].quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
    ejes[1].set_xlabel("w")
    ejes[1].set_ylabel("b")


def en_limites(a,b,xlim,ylim):
    xlow,xhigh = xlim
    ylow,yhigh = ylim
    ax, ay = a
    bx, by = b
    if (ax > xlow and ax < xhigh) and (bx > xlow and bx < xhigh) \
        and (ay > ylow and ay < yhigh) and (by > ylow and by < yhigh):
        return True
    return False

def graficar_contorno_con_descenso(x, y, historial, eje, 
                                   funcion_calcular_costo,
                                   rango_w=[-50, 400, 5], 
                                   rango_b=[-450, 450, 5],
                                   niveles_contorno=[0.1, 100, 1000, 5000, 10000, 25000, 50000],
                                   resolucion=5, w_final=200, b_final=100, paso=10):
    """
    Dibuja el gráfico de contorno de la función de costo J(w, b) y el camino seguido por el descenso del gradiente.

    Parámetros:
        x, y (ndarray): Datos y etiquetas.
        historial (list): Lista con el historial de parámetros [w, b].
        eje (matplotlib axis): Eje donde se dibujará el gráfico.
        rango_w, rango_b (list): Rango para los valores de w y b.
        niveles_contorno (list): Niveles de la función de costo para las líneas de contorno.
        resolucion (int): Distancia mínima entre flechas para el camino.
        w_final, b_final (float): Últimos valores alcanzados por w y b.
        paso (int): Intervalo entre puntos para dibujar las flechas del camino.
    """

    b_vals, w_vals = np.meshgrid(np.arange(*rango_b), np.arange(*rango_w))
    z = np.zeros_like(b_vals)

    for i in range(w_vals.shape[0]):
        for j in range(w_vals.shape[1]):
            z[i][j] = funcion_calcular_costo(x, y, w_vals[i][j], b_vals[i][j])

    contornos = eje.contour(w_vals, b_vals, z, niveles_contorno, linewidths=2)
    eje.clabel(contornos, inline=True, fmt='%1.0f', fontsize=10)
    eje.set_xlabel("w")
    eje.set_ylabel("b")
    eje.set_title("Gráfico de contorno de J(w, b) con trayectoria del descenso del gradiente")

    # Dibujar trayectoria
    punto_anterior = historial[0]
    for punto in historial[0::paso]:
        distancia = np.sqrt((punto_anterior[0] - punto[0])**2 + (punto_anterior[1] - punto[1])**2)
        if distancia > resolucion or punto == historial[-1]:
            if en_limites(punto, punto_anterior, eje.get_xlim(), eje.get_ylim()):
                plt.annotate('', xy=punto, xytext=punto_anterior, xycoords='data',
                             arrowprops={'arrowstyle': '->', 'color': 'b', 'lw': 3},
                             va='center', ha='center')
            punto_anterior = punto
    return