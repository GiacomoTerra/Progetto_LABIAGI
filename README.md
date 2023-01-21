# Progetto_LABIAGI
Programma in Python sulla classificazione e conteggio dei mezzi che passano su una strada trafficata.

Il progetto è scritto in python e con le librerie OpenCv, NumPy e Pandas.
Lo script prende in input un video(in questo contesto ho usato un .mp4) e restituisce in output un secondo video rielaborato.

## Per contare i mezzi all'interno del video il programma esegue una serie di trasformazioni all' immagine:
* Per prima cosa converte l'immagine in scala di grigio
* Applica la Background Subtraction per provare a distinguere gli ogetti in movimento
!(https://149695847.v2.pressablecdn.com/wp-content/uploads/2021/08/input.png)
  
