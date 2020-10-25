# ipln-lab-2020
# Secciones de prueba
Para ejecutar el código debemos ejecutar solamente python3 es_odio.py
En el archivo utils.py, en la función test, se encuentran comentadas distintas pruebas que se ejecutaron para intentar manejar los embeddings, utilizando algunas librerías útiles para su manejo. Igualmente todo lo comentado no creo que sea de mucha utilidad.

# Algunas funciones que pueden ser utiles
La función readLines carga simplemente el corpus de tweets train.csv, y lo lee tweet por tweet (linea por linea), y en cada iteración obtiene la clasificación del tweet. 

La funcion loadVectors carga el archivo de vectores "fasttext.es.300.txt" brindado por los docentes.

En la sección comentada, se utiliza la librería propia de fasttext, que descarga el mismo archivo de vectores "fasttext.es.300" pero con un formato distinto, con el cual se pueden usar las distintas funcionalidades que provee la librería.
Estaría bueno utilizar las funciones que la librería ofrece para manejar los vectores.

# Seccion keras, sección a utilizar 
Aqui tenemos una red neuronal implementada con keras, utilizando la capa de embeddings brindada por los docentes
La misma ejecutando el siguiente comando: python3 example.py