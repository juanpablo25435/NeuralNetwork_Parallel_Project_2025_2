import matplotlib.pyplot as plt
import numpy as np

# DATOS REALES (Extraídos de tus logs)

# 1. Tiempos Totales (10 Epochs)
tiempos_total = {
    'Python MP': 1716.08,
    'C Secuencial': 391.83,
    'OpenMP (1 hilo)': 407.88, # Más lento por overhead
    'OpenMP (4 hilos)': 203.50, # Tu mejor tiempo CPU
    'Python NumPy': 99.56,
    'GPU CUDA': 0.63
}

# 2. Datos para Ley de Amdahl (OpenMP)
# Usamos C Secuencial (391.83) como base de comparación
hilos_lista = [1, 2, 4, 8]
tiempos_omp = [407.88, 271.21, 203.50, 208.14]

# 3. Datos Profiling CUDA (Batch 512 - El eficiente)
# CPU->GPU: 0.562 ms, Kernel: 7.425 ms, GPU->CPU: 0.794 ms
profiling_labels = ['H2D (CPU->GPU)', 'Kernel (Cómputo)', 'D2H (GPU->CPU)']
profiling_data = [0.562, 7.425, 0.794]

# 4. Datos Batch Size (Tiempo por Epoch)
batch_labels = ['Batch 16 (Pequeño)', 'Batch 512 (Grande)']
batch_times = [30.79, 1.03]

# Configuración general de estilo
plt.rcParams.update({'font.size': 12})

# GRÁFICA 1: Comparativa General
def graph_comparativa():
    nombres = list(tiempos_total.keys())
    valores = list(tiempos_total.values())
    
    # Colores: Rojo (Lento) -> Verde (Rápido)
    colores = ['#c0392b', '#d35400', '#e67e22', '#f39c12', '#2980b9', '#27ae60']
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(nombres, valores, color=colores, edgecolor='black')
    
    plt.yscale('log') # ESCALA LOGARÍTMICA
    plt.title('Tiempo Total de Entrenamiento (10 Epochs) - Escala Logarítmica', fontweight='bold')
    plt.ylabel('Tiempo (Segundos)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Etiquetas
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.1, f'{yval:.2f}s', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Flecha de CUDA
    plt.annotate('¡GPU: 0.63s!', xy=(5, 0.63), xytext=(4, 10),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('1_comparativa_tiempos.png', dpi=300)
    print("Generada: 1_comparativa_tiempos.png")

# GRÁFICA 2: Speedup OpenMP y Amdahl
def graph_amdahl():
    # Calculamos Speedup: Tiempo Base (C Seq) / Tiempo Paralelo
    base = 391.83
    speedups = [base / t for t in tiempos_omp]
    
    plt.figure(figsize=(10, 6))
    
    # Línea Ideal (Lineal)
    plt.plot(hilos_lista, hilos_lista, 'k--', label='Speedup Ideal', alpha=0.5)
    
    # Línea Real
    plt.plot(hilos_lista, speedups, 'o-', color='#e74c3c', linewidth=3, markersize=8, label='Speedup Real')
    
    plt.title('Análisis de Escalabilidad OpenMP (Ley de Amdahl)', fontweight='bold')
    plt.xlabel('Número de Hilos')
    plt.ylabel('Speedup (x veces más rápido)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Anotaciones
    for x, y in zip(hilos_lista, speedups):
        plt.text(x, y + 0.2, f'{y:.2f}x', ha='center', color='#c0392b', fontweight='bold')

    plt.tight_layout()
    plt.savefig('2_speedup_amdahl.png', dpi=300)
    print("Generada: 2_speedup_amdahl.png")

# GRÁFICA 3: Profiling CUDA
def graph_profiling():
    # Gráfico de Pastel (Pie Chart)
    plt.figure(figsize=(8, 6))
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    explode = (0.1, 0, 0.1)  # Separar las transferencias
    
    plt.pie(profiling_data, labels=profiling_labels, autopct='%1.1f%%', startangle=140, 
            colors=colors, explode=explode, shadow=True)
    
    plt.title('Desglose de Tiempo en GPU (Batch 512)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('3_profiling_cuda.png', dpi=300)
    print("Generada: 3_profiling_cuda.png")

# GRÁFICA 4: Análisis Batch Size
def graph_batch():
    plt.figure(figsize=(8, 6))
    colors = ['#e74c3c', '#27ae60'] # Rojo malo, Verde bueno
    
    bars = plt.bar(batch_labels, batch_times, color=colors, edgecolor='black', width=0.6)
    
    plt.title('Impacto del Batch Size (Tiempo por Epoch)', fontweight='bold')
    plt.ylabel('Segundos (Menor es mejor)')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}s', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.annotate('30x más rápido', xy=(1, 1.03), xytext=(0.5, 15),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

    plt.tight_layout()
    plt.savefig('4_batch_size_impact.png', dpi=300)
    print("Generada: 4_batch_size_impact.png")

# Ejecutar todo
if __name__ == "__main__":
    graph_comparativa()
    graph_amdahl()
    graph_profiling()
    graph_batch()