# readme seccion redes neuronales
En la carpeta rnn tenemos varios ficheros.py que hacen posible la ejecución de 2 redes neuronales recurrentes distintas. 
Ambas redes entrenan con el corpus "corpus_espanol.txt", de la carpeta resources

# Red neuronal recurrente simple
La primera de ellas es char_rnn_simple.py, y todo el código de la red esta autocontenido en ese archivo. 
Para ejecutar la misma, solo hace falta correr el siguiente comando:
python3 char_rnn_simple.py

# Red neuronal recurrente LSTM 
La segunda red necesita todos los otros archivos para ejecutar correctamente, 
y se puede correr la misma ejecutando el siguiente comando:
python3 main.py

# Seccion Keras
Aqui tenemos una red neuronal implementada con keras, utilizando la capa de embeddings brindada por los docentes
La misma ejecutando el siguiente comando: python3 example.py