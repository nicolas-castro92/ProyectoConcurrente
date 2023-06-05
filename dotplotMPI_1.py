from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
from Bio import SeqIO

begin = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cant_muestras = 150

# Cargar las secuencias desde los archivos .fna
secuencia1 = SeqIO.parse("/home/nico/Documents/Universidad/2023-1/Concurrente/SesiónDotplotConcurrente/archivosParaDotplot/E_coli.fna", "fasta").__next__().seq
secuencia2 = SeqIO.parse("/home/nico/Documents/Universidad/2023-1/Concurrente/SesiónDotplotConcurrente/archivosParaDotplot/Salmonella.fna", "fasta").__next__().seq

# Dividir la secuencia1 en chunks, uno por cada proceso.
chunks = np.array_split(range(len(secuencia1)), size)

dotplot = np.empty([len(chunks[rank]), len(secuencia2)], dtype=np.int32)

for i in range(len(chunks[rank])):
    for j in range(len(secuencia2)):
        if secuencia1[chunks[rank][i]] == secuencia2[j]:
            dotplot[i, j] = np.int32(1)
        else:
            dotplot[i, j] = np.int32(0)

# gather data from all processes onto the root process
dotplot = comm.gather(dotplot, root=0)

# The root process prints the results and generates the plot.
if rank == 0:
    # merge the gathered data into a single array
    merged_data = np.vstack(dotplot)

    end = time.time()
    print(f"Tiempo total de ejecución: {end - begin} segundos")

    plt.figure(figsize=(10, 10))
    plt.imshow(merged_data[:500, :500], cmap='Greys', aspect='auto')
    plt.savefig(f"ResultadoMPI_{size}.png")
