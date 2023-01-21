# Progetto_LABIAGI
Programma in Python sulla classificazione e conteggio dei mezzi che passano su una strada trafficata.

Il progetto è scritto in python e con le librerie OpenCv, NumPy e Pandas.
Lo script prende in input un video(in questo contesto ho usato un .mp4) e restituisce in output un secondo video rielaborato.

## Per contare i mezzi all'interno del video il programma esegue una serie di trasformazioni all' immagine:
* Per prima cosa converte l'immagine in scala di grigio
* Applica la Background Subtraction per provare a distinguere gli ogetti in movimento
![esempio](https://149695847.v2.pressablecdn.com/wp-content/uploads/2021/08/input.png)
![esempio](https://149695847.v2.pressablecdn.com/wp-content/uploads/2021/08/MOG2.png)
* Dopo vengono rimosse le ombre ed eventuali disturbi grandi applicando alcune trasformazioni morfologiche
 https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
* Una volta isolate le macchine vengono disegnati i contorni attorno ad esse il contorno ed il rettangolo
 https://docs.opencv.org/3.4/d7/d1d/tutorial_hull.html
 https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
* Ed infine vengono disegnate le 2 linee per il conteggio 

