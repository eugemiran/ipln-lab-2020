# ipln-lab-2020
Grupo 14 
Integrantes :
Agustina Sierra 4647235-6
Agustina Salmanton xxxxxxx-x
Eugenia Miranda xxxxxxx-x
Sebastian Volti xxxxxxx-x

1-Limpieza de datos
Librerias utilizadas: Pandas version 1.0.3 , Re version 2.2.1 , nltk versiom 3.5

En el archivo data.py se encuentan las diferentes funciones que se utilizaron para limpiar los datos.
Inicialmente se cargaton todos los datos en un Data Frame con dos columnas , una el el tweet y la siguiente su clasificacion.
Luego a cada uno de estos tweets se los sometio a un pipeline de funciones que hacen lo siguiente:

removen(text) => Remueve los caracteres '\n'
removeUrl(text) => Remueve las URL
removeHashtags(text) => Remueve los Hashtags
toLowerCase(text) => Convierte cada carácter en mayúscula a minúscula
removeUsers(text) => Remueve las menciones a usuarios (@user)
removeRepetitions(text) => Remueve las letras repetidas de palabras con letras repetidas y deja una ocurrencia por ejemplo "holaaa" => "hola" 
removePunctuation(text) =>  Remueve cualquier caracter no alfanumerico
removeLaughter(text) => Modifica algunas acepciones de risa por "jaja" por ejemplo "jajajajajaja","jeje","jajsj"

2-Modelos elegidos
Librerias utilizadas: numpy version 1.18.2 , keras version 2.4.3 , tensorflow version 2.3.1 

2.1- Modelo de red neuronal simple
2.2- Redes neuronales LSTM
2.3- Red neuronal Bidireccional
2.4- Red neuronal Convolucional

3- Cross Validation
Para encontrar el mejor modelo con sus mejores parametros , se implemento validacion cruzada la misma se encuentra en el archivo CrossValidation.py y en el archivo CrossValidation.ipynb se encuenta con detalle como se realizo este procedimiento.
Los parametros que se ajustaron aqui son : epochs,neurons,dropout,batchs,model_type.
Se encontro que los mejores paramentros fueron ####


4-Algunas funciones que pueden ser utiles
La función readLines carga simplemente el corpus de tweets train.csv, y lo lee tweet por tweet (linea por linea), y en cada iteración obtiene la clasificación del tweet. 

La funcion loadVectors carga el archivo de vectores "fasttext.es.300.txt" brindado por los docentes.

En la sección comentada, se utiliza la librería propia de fasttext, que descarga el mismo archivo de vectores "fasttext.es.300" pero con un formato distinto, con el cual se pueden usar las distintas funcionalidades que provee la librería.
Estaría bueno utilizar las funciones que la librería ofrece para manejar los vectores.

5-Como correr la tarea
Para ejecutar el código debemos ejecutar solamente python3 es_odio.py
