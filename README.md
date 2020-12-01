# ipln-lab-2020
Grupo 14 integrantes :
- Agustina Sierra 4.647.235-6
- Agustina Salmanton 4.877.872-8
- Maria Eugenia Miranda 4.774.499-8
- Sebastian Volti 5.175.914-7

## Ejecucion 
Para la ejecución hay que instalar los requerimientos mediante el comando:

`pip3 install -r requirements.txt`

Para la ejecucion del archivo principal, se respeta el formato pedido:

`python3 es_odio.py <data_path> test_file1.csv ... test_fileN.csv`

Para la ejecucion del modelo con el archivo *test.csv* recibido como resource:

`python3 es_odio.py ./resources`

Para la ejecución del cross validation, se debe ejecutar el siguiente comando:

`python3 crossValidation.py`


## Archivos
 Se incluyen los siguientes archivos:
 - **es_odio.py**: archivo principal, donde se crea un modelo, se lo entrena y evalua
 - **model.py**: archivo donde se encuentran los modelos implementados
 - **data.py**: archivo donde se cargan y limpian los datos
 - **constants.py**: archivo donde se tienen constantes generales
 - **crossValidation.py**: ejecucion de la configuracion de los hiperparametros mediante cross validation.
 
## Limpieza de datos
Librerias utilizadas:
- pandas v1.0.3
- nltk v3.5

En el archivo **data.py** se encuentan las diferentes funciones que se utilizaron para limpiar los datos.
Inicialmente se cargaron todos los datos en un DataFrame con dos columnas, una para los tweets y la otra para su clasificacion.
Luego a cada uno de estos tweets se los sometio a un pipeline de funciones de preprocesamiento:

- **removen(text)**: Remueve los caracteres '\n'
- **removeUrl(text)**: Remueve cambia las url por el string URL
- **removeHashtags(text)**: Remueve los simbolos de hashtag (#)
- **toLowerCase(text)**: Convierte cada carácter en mayúscula a minúscula
- **removeUsers(text)**: Remueve las menciones a usuarios (@user) y las cambias por el string USER
- **removeRepetitions(text)**: Remueve las letras repetidas de palabras con letras repetidas y deja una ocurrencia (**i.e**: "holaaa" => "hola")
- **removePunctuation(text)**: Remueve cualquier caracter no alfanumerico
- **removeLaughter(text)**: Modifica algunas acepciones de risa por "jaja" (**i.e**: "jajajajajaja"=> "jaja", "jeje" => "jaja", "jajsj" => "jaja")

## Modelos elegidos
Librerias utilizadas:
- numpy v1.18.2
- keras v2.4.3
- tensorflow v2.3.1 

En el archivo **model.py** se encuentra definida la clase Model la cual crea el modelo y contienen las diferentes funciones que es de interes aplicar sobre los modelos. Los modelos tienen metodos de entrenamiento y evaluacion, estos son **eval()** y **train()** respectivamente.
Para inicializar los modelos utilizamos los embbeding provistos. Inicialmente toquenizamos los tweets y generamos una matriz de embbedings, donde por cada palabra del volcabulario encontrada colocamos su vector de embedings como columna en la matriz. Luego esa matriz es utilizada como primera capa en nuestras redes neuronales.

### Modelos generados:
Modelo de red neuronal simple.
Redes neuronales LSTM1.
Redes neuronales LSTM2.
Red neuronal Bidireccional.
Red neuronal Convolucional.
En el archivo **CrossValidation.ipynb** se encuenta un detalle de las mismas.

## Cross Validation
Para encontrar el mejor modelo con sus mejores parametros, se implementó validación cruzada, la misma se encuentra en el archivo **CrossValidation.py** y en el archivo **CrossValidation.ipynb** se encuenta con detalle como se realizo este procedimiento.
Los parametros que se ajustaron aqui son:
- epochs
- neurons
- dropout
- batchs
- model_type.

Se concluyo que el mejor modelo es el Modelo Convolutional.

A su vez, los mejores paramentros fueron:
|parameter|value|
|---|---|
|dropout|0.1|
|epocas|10|
|neuronas|64|
|batches|64|


## Resultados Obtenidos :
Ejecutando **es_odio.py** con el conjunto de entrenamiento **test.csv** se obtuvieron los siguientes resultados:

|Metrica|Valor|
|---|---|
|Accuracy|55.14|
|Precision|57.63|
|Recall|45.09|
|F1|49.28|
