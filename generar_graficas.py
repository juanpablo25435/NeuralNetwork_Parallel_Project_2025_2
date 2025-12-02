import matplotlib.pyplot as plt
import numpy as np

# --- DATOS REALES DEL PROYECTO ---
# Tiempos normalizados a 10 Epochs
# Nota: Python MP tardó 858s en 5 epochs -> Proyectamos 1716s para 10 epochs
datos = {
    'C Secuencial': 391.83,
    'OpenMP (8 Hilos)': 229.92,
    'Python (NumPy)': 99.56,
    'Python MP (10 Ep*)': 1716.08,
    'GPU (CUDA)': 0.63
}

# Configuración de estilo
plt.style.use('default') # Estilo limpio

# ==========================================
# GRÁFICA 1: COMPARATIVA TOTAL DE TIEMPOS
# ==========================================
def plot_total_times():
    nombres = list(datos.keys())
    valores = list(datos.values())
    
    # Colores semánticos:
    # Rojo=Lento, Amarillo=Medio, Verde=Rápido, Azul=Base C
    colores = ['#7f8c8d', '#e67e22', '#f1c40f', '#c0392b', '#2ecc71'] 

    fig, ax = plt.subplots(figsize=(12, 7))
    barras = ax.bar(nombres, valores, color=colores, edgecolor='black', alpha=0.9)

    # ESCALA LOGARÍTMICA: Fundamental para ver la barra de CUDA
    ax.set_yscale('log')
    
    # Títulos y Etiquetas
    ax.set_title('Tiempo de Entrenamiento por Implementación (10 Epochs)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Tiempo (Segundos) - Escala Logarítmica', fontsize=12)
    ax.set_xlabel('Implementación', fontsize=12)
    
    # Grid suave
    ax.grid(axis='y', linestyle='--', alpha=0.5, which='major')

    # Poner el valor exacto encima de cada barra
    for barra in barras:
        height = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., height * 1.1,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Anotación especial para CUDA
    speedup_cuda = datos['C Secuencial'] / datos['GPU (CUDA)']
    ax.annotate(f'¡Speedup {speedup_cuda:.0f}x!', 
                xy=(4, 0.63), xytext=(3, 3),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('grafica_comparativa_log.png', dpi=300)
    print(">> Generada: grafica_comparativa_log.png")
    plt.show()

# ==========================================
# GRÁFICA 2: SPEEDUP OPENMP (LEY DE AMDAHL)
# ==========================================
def plot_speedup():
    # Datos: 1 Hilo (Base) vs 8 Hilos (Tu ejecución)
    hilos = [1, 8]
    
    t_seq = 391.83
    t_omp = 229.92
    
    speedup_real = t_seq / t_omp # 1.70x
    speedups = [1.0, speedup_real]
    
    # Ideal teórico (Lineal)
    ideal = [1.0, 8.0]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Línea Ideal
    ax.plot(hilos, ideal, 'k--', label='Speedup Ideal (Lineal)', alpha=0.5)
    
    # Línea Real
    ax.plot(hilos, speedups, 'o-', color='#e67e22', linewidth=3, markersize=10, label='Speedup Real (OpenMP)')
    
    # Rellenar el área de "Pérdida por Eficiencia"
    ax.fill_between(hilos, speedups, ideal, color='gray', alpha=0.1, label='Pérdida (Overhead/Amdahl)')

    # Etiquetas
    ax.set_title('Análisis de Escalabilidad OpenMP (8 Hilos)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Número de Hilos de CPU', fontsize=12)
    ax.set_ylabel('Speedup (Veces más rápido)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Límites
    ax.set_ylim(0, 9)
    ax.set_xlim(0, 9)

    # Anotación del valor
    ax.annotate(f'{speedup_real:.2f}x', 
                 xy=(8, speedup_real), xytext=(6, speedup_real + 1),
                 arrowprops=dict(facecolor='#e67e22', shrink=0.05),
                 fontsize=14, fontweight='bold', color='#e67e22')

    plt.tight_layout()
    plt.savefig('grafica_speedup_amdahl.png', dpi=300)
    print(">> Generada: grafica_speedup_amdahl.png")
    plt.show()

if __name__ == "__main__":
    plot_total_times()
    plot_speedup()