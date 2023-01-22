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
 
 Dilation
 It is just opposite of erosion. Here, a pixel element is '1' if at least one pixel under the kernel is '1'. So it increases the white region in the image or size of     foreground object increases. Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won't come back, but our object area increases. It is also useful in joining broken parts of an object.
dilation = cv.dilate(img,kernel,iterations = 1)
![](https://docs.opencv.org/3.4/dilation.png)

Opening
Opening is just another name of erosion followed by dilation. It is useful in removing noise, as we explained above. Here we use the function,
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
![](https://docs.opencv.org/3.4/opening.png)

Closing
Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
![](https://docs.opencv.org/3.4/closing.png)
* Una volta isolate le macchine vengono disegnati i contorni attorno ad esse il contorno ed il rettangolo
 https://docs.opencv.org/3.4/d7/d1d/tutorial_hull.html
 https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
* Ed infine vengono disegnate le 2 linee per il conteggio:
  la linea per il conteggio è una, l'altra serve per determinare la direzione

Tutte le identificazioni vengono salvate su un Data Frame(Pandas):
ogni riga è un frame differente mentre ad ogni nuovo id viene aggiunta una nuova colonna 
ogni sezione avrà le coordinate (x,y) di quel car id nel determinato frame
Il Data Frame viene salvato al termine in un file csv

