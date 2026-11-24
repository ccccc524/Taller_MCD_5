#######################################################################################
#######################################################################################
################      CLASE 12                      ###################################
################  MCD Carlos Castro Correa          ###################################
#######################################################################################
#######################################################################################

#Contenido: IA Generativa + NLP + Modelos de lenguaje 

#Paquetes hasta el momento:
library(readr)
library(tidyverse)
library(leaflet)
library(leaflet.extras)
library(rpart) #arboles de decisón
library(rpart.plot) #graficos de arbol de decision
library(readxl) #leer archivos de excel
#library(xgboost) #algoritmo para paralelizar arboles de decision
detach("package:xgboost", unload = TRUE) #Quitamos de la sesión a xgoboost pues tiene conflicto con la función slice()
library(tidymodels)
library(caret)
library(lubridate)#Paquete para lidiar con tiempos y fechas de procesamiento
library(rpart)
library(caret)

#Paquete nuevo a instalar
library(tidytext)
library(stringdist) 
library(bayesbio)



#Primero leemos los diversos n-gramas creados a partir de millones de artículos en español
#Datos
setwd("~/Desktop/Carlos/cursos/ITAM/Diplomado_ML_noviembre2026/Clase_7_8")

#Caracteres a retirar de los textos
quitar <- read_csv("quitar.csv")

#Muestra de los articulos con los que fueron creados
sample <- read_csv("sample_news.csv")

sample$txt[1]

sample$txt[30]

sample$txt[99]



#######################################################################################
################      Limpieza                     ####################################

#Para mejorar la precision del análisis de texto podemos utilizar funciones para limpiar y organizar texto

normalizar <- function(texto, vocab = NULL){
  # minúsculas
  texto <- tolower(texto)
  # varios ajustes
  texto <- gsub("\\s+", " ", texto)
  texto <- gsub("\\.[^0-9]", " _punto_ ", texto)
  texto <- gsub(" _s_ $", "", texto)
  texto <- gsub("\\.", " _punto_ ", texto)
  texto <- gsub("[«»¡!¿?-]", "", texto) 
  texto <- gsub(";", " _punto_coma_ ", texto) 
  texto <- gsub("\\:", " _dos_puntos_ ", texto) 
  texto <- gsub("\\,[^0-9]", " _coma_ ", texto)
  texto <- gsub("\\s+", " ", texto)
  texto
}

#Ejemplo limpieza
corpus <- c("Lista de las comidas: pollo en chile, frijoles, hummus; 
                 no es posible incluir comida japones. No es verdad, 
                 todo lo que estás dicienciendo .
                 El máximo de participantes de las competencias es 10, 
                 no es posible iniciar con menos de 5 jugadores",
            "Finalmente; este es otro texto en donde; 
                 se usa indecuadamente punto y coma " )

corpus

#Aplicamos normalización de texto
normalizar(corpus)


#Para crear algunos n-gramas con el corpus podemos utilizar el siguiente proceso

ejemplo <- tibble(txt = corpus) %>%
  mutate(id = row_number()) %>%
  mutate(txt = normalizar(txt)) 

#Bi-gramas
bigrams_ejemplo <- ejemplo %>% 
  unnest_tokens(bigramas, txt, token = "ngrams", 
                n = 2) %>%
  group_by(bigramas) %>% tally() %>% 
  arrange(desc(n)) %>% View()

#¿En donde aparece "no es"?

#¿En donde aparece "de las"?

#¿En donde aparece "es posible"?

#¿Qué sucede con el resto de los bigramas?...




#La función "normalizar" es aplicada a la base de datos del periodico y creamos los siguientes conjuntos:

#######################################################################################
################      n-gramas                     ####################################

#Unigrama
mod_uni <- read_csv("unigramas.csv")

#Bigrama
mod_bi <- read_csv("bigramas.csv")

#Trigrama
setwd("~/Desktop/Carlos/cursos/ITAM/Diplomado_ML_noviembre2025/Clase6") #Demasiado pesado, lo dejo en carpeta anterior
mod_tri <- read_csv("trigramas.csv")


#Observa el tamaño de nuestros n-gramas:

dim(mod_uni)

dim(mod_bi)

dim(mod_tri)


#Observa algunos de los registros

View(mod_uni)

View(mod_bi)

View(mod_tri)


#¿Cómo podemos hacer que una máquina produzca texto?...

#######################################################################################
################      Modelos de lenguaje                     #########################


#Los resultados de este modelo se basan en los resultados de los bi-gramas

#Palabra inicial
palabra <- "abogados"
#Frase construida a partir de la suceción de diversas palabras
frase <- palabra #Al inicio la frase y palabra son la misma (solo tenemos una)

#¿Cómmo funciona nuestra función para construir frases?...

#En el modelo de bi-grama buscamos en donde ocurrió la palabra...
#y ordenamos para encontrar la palabra más probable que le sigue..

#Nota: sobre los bi-gramas que acabamos de crear (entrenar), se buscan estas coincidencias:

mod_bi %>%  
  filter(!w_n_1 %in% quitar$quitar) %>%
  filter(!w_n_0 %in% quitar$quitar) %>% 
  filter(w_n_1 == palabra) %>% 
  arrange(desc(num)) %>% View()
#Nuestro modelo va a asignar la siguiente palabra con más observaciones...


#Nota: Utilizamos una función para construir frases:

pred_chat <- function(palabra,frase){
  sig <- mod_bi %>%  #En el modelo de bi-grama buscamos en donde ocurrió la palabra...
    filter(!w_n_1 %in% quitar$quitar) %>%
    filter(!w_n_0 %in% quitar$quitar) %>%
    filter(w_n_1 == palabra) %>% 
    arrange(desc(num)) %>% 
    select(w_n_0) %>% slice(1)
  #Pegamos para la frase
  frase <- paste(frase,sig$w_n_0,sep=' ')
  return(list(frase,sig))
}

#Al inicio palabra y frase son las mismas
# ¿Por qué?
palabra

frase

pal <- pred_chat(palabra,frase)

pal #lista con 2 salidas

#Frase
pal[[1]][1]

#sig
pal[[2]][1]

#Pregunta:
#¿Qué está pasando realmente?
#¿Existe un fundamento matemático para constuir frases de esta forma?...


#De todas las posibles frases, está eligiendo la más probable...


















#Regresamos a las notas...

#Creador de frases
#Utilizamos iterativamente esta función para crear frase con longitud 3
palabra <- "abogados"
frase <- palabra

#Construcción inicial
#Longitud 3
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:1){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]

#Ahora toma la última palabra generada y entra de nuevo en el algoritmo:
pal[[2]][1]


#Longitud 4
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:2){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]

#Ahora toma la última palabra generada y entra de nuevo en el algoritmo:
pal[[2]][1]



#Longitud 5
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:3){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]

#Ahora toma la última palabra generada y entra de nuevo en el algoritmo:
pal[[2]][1]


#Longitud 6
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:4){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]

#Ahora toma la última palabra generada y entra de nuevo en el algoritmo:
pal[[2]][1]



#Longitud 7
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:5){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]



#Longitud 8
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:6){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]


#Longitud 9
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:7){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]


#Longitud 10
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:8){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]


#¿Que va a pasar con la siguiente iteración?....









#Longitud 11
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:9){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]


#Longitud 12
pal <- pred_chat("abogados","abogados") #Primero va la palabra y luego la frase
for(i in 1:10){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]


#...
#...
#Además, ¿cómo funciona con otras palabras?

pal <- pred_chat("hola","hola") #Primero va la palabra y luego la frase
for(i in 1:8){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]



#Además, ¿cómo funciona con otras palabras?
pal <- pred_chat("griegos","griegos") #Primero va la palabra y luego la frase
for(i in 1:8){
  pal <- pred_chat(as.character(pal[[2]][1]),as.character(pal[[1]][1]))
}
#Frase
pal[[1]][1]


#Preguntas:
#¿Qué esta sucediendo con nuestro modelo?
#Gran pregunta: ¿Cómo podrías mejorarlo?...





















#######################################################################################
################      Modelos con temperatura (random)      ###########################


#Modelo random
palabra <- "deporte"
frase <- palabra
temperatura <-0.8 #valor entre 0 y 1 (sin tocar los extremos)

#Ajustamos la funcion pred_chat para asignar aleatoriamente entre cierto número de registros
pred_chat_temperatura <- function(palabra,frase,temperatura){
  sig <- mod_bi %>%
    filter(!w_n_1 %in% quitar$quitar) %>%
    filter(!w_n_0 %in% quitar$quitar) %>%
    filter(w_n_1 == palabra) %>% 
    arrange(desc(num)) 
  #Matriz auxiliar para muestreo
  sig_bis <- sig %>% arrange((num))
  
  #Nota el ajuste en TEMPERATURA
  
  #Siguiente palabra con Temperatura
  sig_bis <- sig_bis[1:floor(nrow(sig)*(1-temperatura)),"w_n_0"] %>% 
    sample_n(size = 1)
  #Pegamos para la frase
  frase <- paste(frase,sig_bis$w_n_0,sep=' ')
  return(list(frase,sig_bis$w_n_0))
}



palabra <- "abogados"
frase <- palabra

#Longitud 3
pal <- pred_chat_temperatura("abogados","abogados",temperatura) #Primero va la palabra y luego la frase
for(i in 1:1){
  pal <- pred_chat_temperatura(as.character(pal[[2]][1]),as.character(pal[[1]][1]),temperatura)
}
#Frase
pal[[1]][1]



###Además cada vez que llamamos la función pred_chat_temperatura()
#Entrega un resultado distintos y con mayor coherencia...

#Longitud 2
for(k in 1:10){
  pal <- pred_chat_temperatura("abogados","abogados",temperatura) #Primero va la palabra y luego la frase
  for(i in 1:1){
    pal <- pred_chat_temperatura(as.character(pal[[2]][1]),as.character(pal[[1]][1]),temperatura)
    print(pal[[1]][1])
  }
}


#Aumentamos la longitud
#Longitud 6
for(k in 1:10){
  pal <- pred_chat_temperatura("abogados","abogados",temperatura) #Primero va la palabra y luego la frase
  for(i in 1:5){
    pal <- pred_chat_temperatura(as.character(pal[[2]][1]),as.character(pal[[1]][1]),temperatura)
  }
  print(pal[[1]][1])
}



#Aumentamos aún más la longitud
for(k in 1:15){
  pal <- pred_chat_temperatura("abogados","abogados",temperatura) #Primero va la palabra y luego la frase
  for(i in 1:5){
    pal <- pred_chat_temperatura(as.character(pal[[2]][1]),as.character(pal[[1]][1]),temperatura)
  }
  print(pal[[1]][1])
}




#Cambiamos la temperatura

palabra <- "abogados"
frase <- palabra
temperatura <-0.8 #valor entre 0 y 1 (sin tocar los extremos)

for(k in 1:10){
  pal <- pred_chat_temperatura(palabra,frase,temperatura) #Primero va la palabra y luego la frase
  for(i in 1:5){
    pal <- pred_chat_temperatura(as.character(pal[[2]][1]),as.character(pal[[1]][1]),temperatura)
  }
  print(pal[[1]][1])
}


#Temperatura 0.3
palabra <- "abogados"
frase <- palabra
temperatura <-0.3 #valor entre 0 y 1 (sin tocar los extremos)

for(k in 1:10){
  pal <- pred_chat_temperatura(palabra,frase,temperatura) #Primero va la palabra y luego la frase
  for(i in 1:5){
    pal <- pred_chat_temperatura(as.character(pal[[2]][1]),as.character(pal[[1]][1]),temperatura)
  }
  print(pal[[1]][1])
}



#Temperatura 0.9
palabra <- "abogados"
frase <- palabra
temperatura <-0.9 #valor entre 0 y 1 (sin tocar los extremos)

for(k in 1:10){
  pal <- pred_chat_temperatura(palabra,frase,temperatura) #Primero va la palabra y luego la frase
  for(i in 1:5){
    pal <- pred_chat_temperatura(as.character(pal[[2]][1]),as.character(pal[[1]][1]),temperatura)
  }
  print(pal[[1]][1])
}

#Pregunta:
#¿Alguien utilizó las primeras versión de ChatGPT?... 
#producía un texto con un nivel de coherencia similar a este ejemplo...

#Notemos que:
#Entre más longitud, menos coherencia (¿por qué?)
#¿Cómo usarías la noción de conjunto de entrenamiento para explicar este hecho?

#El modelo con temperatura (random) funciona mejor mediante una decisión que no parecía evidente

#Importancia de lo aleatorio
#Ejemplo: descenso en gradiente
#Por ejmplo, para optimizar funciones (PARÁMETROS ÓPTIMOS EN ML)...
#https://es.wikipedia.org/wiki/Descenso_del_gradiente









#######################################################################################
#######################################################################################
################                    NLP              ##################################
#######################################################################################
#######################################################################################

library(tm)
library(topicmodels)


#######################################################################################
################      Índice de Jaccard       #########################################

#Oraciones del ejemplo
String_A <- c("Hola","lo","siento","pero","no","puedo","asistir","hoy")

String_B <- c("Hola","lo","entiendo","pero","no","puedo","escuchar","hoy")

#Cadenas de texto
String_A

String_B


#Intersección - como notación de conjuntos
intersect(String_A,String_B) 

intersect(String_A,String_B) %>% length() 


#Union - como notación de conjuntos
union(String_A,String_B) 

union(String_A,String_B) %>% length()


#Índice de Jaccard
(intersect(String_A,String_B) %>% length() )  /  (union(String_A,String_B) %>% length())


#Alternativa
#También podemos utilizar una función del paquete "bayesbio"
jaccardSets(String_A, String_B) 













#######################################################################################
#######################################################################################
################    #Latent Dirichlet Allocation      #################################
#######################################################################################
#######################################################################################

#Leemos una base original de artículos
# ag <- read_csv(
#   "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
#   col_names = c("class","title","description")
# )
# 
# #Observa los datos en el enlace:
# #https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
# 
# acq <- sample_n(tbl = ag,size = 1000) %>%
#   select(description)

#Mejor cargamos una base para que la muestra no cambie
setwd("~/Desktop/Carlos/cursos/ITAM/Diplomado_ML_noviembre2026/Clase_7_8")
acq <- read_csv("muestra.csv")

#Observamos los datos
View(acq)

#Únicamente tomamos una m.a.
dim(acq)



#Formato:
#Convertir a funcion corpus
acq <- VCorpus(VectorSource(acq$description))

inspect(acq[[1]])

inspect(acq[[2]])

inspect(acq[[20]])

inspect(acq[[24]])

inspect(acq[[40]])

inspect(acq[[44]])

inspect(acq[[49]])


#Guardamos todos estos datos en la variable corpus
corpus <- acq

#Mismos datos
inspect(corpus[[1]])


#######################################################################################
#Transformación de datos
#Raecuerda: el análisis de texto funciona mejor cuando la información esta "limpia y estandarizada"

#Convierto a minúsculas
corpus <- tm_map(corpus, content_transformer(tolower))
inspect(corpus[[1]])

#Remuevo todos los signos de puntuacion:
corpus <- tm_map(corpus, removePunctuation)
inspect(corpus[[1]])

#Remuevo los números
corpus <- tm_map(corpus, removeNumbers)
inspect(corpus[[1]])

#Remuevo las "stop words"
#¿Qué es una stop word?... palabras atípicamente frecuentes que pueden sesgar los resultados
corpus <- tm_map(corpus, removeWords, stopwords("english"))
inspect(corpus[[1]])

#Acomodo el texto
corpus <- tm_map(corpus, stripWhitespace)
inspect(corpus[[1]])

#Creo una matriz de "términos" para el modelo LDA
dtm <- DocumentTermMatrix(corpus)

dtm


#Tamaño de la matriz
dim(dtm)


#Revisamos las entradas de la matriz:
inspect(dtm[1:20, 1:200]) %>% View()
#¿Qué está pasando?...

#Por cada palabra única, indica con 0 ó 1 si esa noticia contiene dicha palabra o no...


#Convertimos la matriz de términos a formato matriz
m <- as.matrix(dtm)

#Observa la matriz y ordena algunas columnas
View(m)

#Palabras de la primer noticia
m[1, m[1, ] > 0] %>% sort()


#Palabras de la segunda noticia
m[2, m[2, ] > 0] %>% sort()

#Pregunta, con la evidencia empírica, ¿de qué crees que trata esta segunda noticia?...


#######################################################################################
################          Algoritmo LDA               #################################
#######################################################################################


#Semilla aleatoria
set.seed(1035)

#Corremos el algoritmo Latent Dirichlet Allocation
#y pedimos que agrupe las noticias en 3 conglomerados:
lda <- LDA(dtm, k = 3,
           control = list(seed = 1035))

#Objeto "poco" informativo
lda

#Temas y términos frecuentes en las noticias de cada grupo:
terms(lda, 10)

#Distribuciones posterior de los términos:
theta <- posterior(lda)$topics
posterior(lda)$topics %>% View()

#Nota: ver en Excel...
#Heatmap matricial...

#Asignamos el tema o topic al que pertenece cada noticia:
tema <- apply(theta, 1, which.max)

tema 
#Observa con atención el paso anterior: de acuerdo a las distribuciones posteriores, se asigna el tema del que trata...

#Convertimos nuevamente a data.frame el objeto de noticias:
acq_df <- data.frame(
  description = sapply(acq, function(x) x$content),
  stringsAsFactors = FALSE
)

#Asignamos cada noticia a un topic o tema
acq_df$tema <- tema

View(acq_df)

#Número de noticias por topic o tema:
table(acq_df$tema) #Grupos equilibrados...


#Algunas noticias del grupo 1:
acq_df %>% 
  filter(tema == "1") %>% 
  sample_n(10) %>% 
  select(description)


#Algunas noticias del grupo 2:
acq_df %>% 
  filter(tema == "2") %>% 
  sample_n(10) %>% 
  select(description)


#Algunas noticias del grupo 3:
acq_df %>% 
  filter(tema == "3") %>% 
  sample_n(10) %>% 
  select(description) 

#¿Te hace sentido la agrupacion de los temas?
#Probablemente 


#Pregunta adicional:
#¿Observas alguna falla en la aplicación del algoritmo LDA?
#¿Qué podríamos agregar?

#¿Qué puedes observar en los parámetros de la función LDA?

#Pregunta importante: ¿Qué tipo de aprendizaje utiliza el algoritmo LDA?
lda <- LDA(dtm, k = 3,
           control = list(seed = 1035))

#Pregunta:
#1. ¿Observas alguna similitud respecto a clustering?
#2. ¿Consideras que estamos haciendo algo como one-hot-encoding y distancia euclidiana?  :((((

#Para responder: lee el siguiente paper: 
#https://web.stanford.edu/class/linguist289/lda.pdf




#Análisis alternativo:

#Dimensión de la matriz:
dim(m)

View(m) #Es una matriz con muchos ceros...

#Registros totales igual a cero
sum(m == 0)

#Porcentaje
sum(m == 0) / (nrow(m)*ncol(m)) #Matriz rala...

#Ya lo sabíamos desde que convertirmos los datos originales a una matriz de términos:
dtm


#¿Qué otro algoritmo de ML podrías utilizar para agrupar las noticias en temas?...







#######################################################################################################
#######################################################################################################
### Supongamos que después de un minucioso análisis... alguien decide usar k-means (ML no supervisado)

#Por supuesto, despúes de leer los paper sobre DLA (hay que estar seguro y entender antes de aplicar cualquier herramienta...)


set.seed(1035)
#Clustering por k-medias con 3 grupos
cluster_3 <- kmeans(m,3)

#Asignamos cada noticia a un grupo
cluster_3$cluster

#Withinss por grupo
cluster_3$withinss

#Whithinss total
cluster_3$tot.withinss

#Ahora, renombramos para reciclar el código de la clase anterior:
datos <- m

#Hacemos distintos cluster para obtener gráfica de codo:

#Cluster con 1 grupos
model_1=kmeans(datos,1)

#Cluster con 2 grupos
model_2=kmeans(datos,2)

#Cluster con 3 grupos
model_3=kmeans(datos,3)

#Cluster con 4 grupos
model_4=kmeans(datos,4)

#Cluster con 5 grupos
model_5=kmeans(datos,5)

#Cluster con 6 grupos
model_6=kmeans(datos,6)

#Cluster con 7 grupos
model_7=kmeans(datos,7)

#Matriz vacía para almacenar los resultados
mat <- matrix(0,7,2) %>% data.frame()
colnames(mat) <- c("cluster","withinss")

#Simplemente lleno mi matriz con los el withinss total de cada modelo
mat$cluster <- c(1,2,3,4,5,6,7)
mat$withinss <- c(model_1$tot.withinss,model_2$tot.withinss,model_3$tot.withinss,model_4$tot.withinss,
                  model_5$tot.withinss,model_6$tot.withinss,model_7$tot.withinss)

#Revisamos la matriz
View(mat)

#Grafica de codo
plot(mat$withinss)
lines(mat$withinss)

#¿Cuál es el número de clusteres o grupos óptimo?


### Pregunta final:
#¿Qué efecto tiene aumentar el número de clústeres o grupos?

table(model_1$cluster)

table(model_2$cluster)

table(model_3$cluster)

table(model_4$cluster)

table(model_5$cluster)

table(model_6$cluster)

#¿Crees que el clustering sea el modelo adecuado para enfrentar este tipo de problemas?
#¿Por qué?
#Meditalo desde el punto de vista geométrico...


#Desde el punto de vista geométrico - matemático, ¿qué está ocurriendo?...
colnames(m)


#Tomemos 3 variables (justo cuando tomamos el ejemplo de iris)...
#chains,captain,baltimore
data <- m[,c("chains", "captain", "baltimore")] %>% 
  data.frame()

View(data)

dim(data)

#Gráficamos en 3D:
plot_ly( 
  data = data,
  x = ~chains, y = ~captain, z = ~baltimore, 
  type = "scatter3d",
  mode = "markers",
  marker = list(color = "royalblue", size = 4)
) %>%
  layout(scene = list(
    xaxis = list(title = "chains"),
    yaxis = list(title = "captain"),
    zaxis = list(title = "baltimore")))

#¿Cuántos registros tienen estas palabras?...

#Pregunta:
#¿En dónde se concentran la mayoría de los registros?

#Registros con chains
which((data[,"chains"] == 1) == "TRUE")

which((data[,"chains"] == 1) == "TRUE") %>%
  length()

#Registros con captain
which((data[,"captain"] == 1) == "TRUE")

which((data[,"captain"] == 1) == "TRUE") %>%
  length()

#Registros con baltimore
which((data[,"baltimore"] == 1) == "TRUE")

which((data[,"baltimore"] == 1) == "TRUE") %>%
  length()


#Los datos quedan sumamente esparcidos (espacialmente hablando)...
#Porque hay muchas palabras UNICAS, que aparecen POCAS veces en los registros...

#IMPORTANTE:
#Además empieza a crear grupos de una observacion 

#Y en algunos cluster, crear grupo de menos observaciones, por ejemplo 10,8, entre otros:

table(model_3$cluster)

table(model_4$cluster)

table(model_5$cluster)

table(model_6$cluster)

table(model_7$cluster)

#El model_7 crea el grupo 5  con 6 observaciones...
table(model_7$cluster)

lista <- which(model_7$cluster == "5") %>% unique()
lista

acq_df[lista,1]

#¿Qué tienen en común estos datos?
#Observemos....

#¿No queda claro? Quizás preguntando cuál es la característica común a un LLM...
#que tema o palabra en comun tienen los sig noticias:

acq_df[lista[3],]

acq_df[lista[6],]

acq_df[lista[4],]

#Observa que la clasificación anterior, la tenía que realizar un humano descartando temas....
#Aunque para esos casos, LDA funciona mejor...

#Lo demás puede ser ruido... aunque podríamos quitar algunos conceptos o stopwords adicionales...


#Nota: por esta razón, en este curos analizamos el fundamento geométrico-matemático de los algoritmos...


#Pregunta final:

#Al parecer aplicando k-means podríamos ir identificando textos con TEMAS SIMILARES, pero...
#¿Es correcto que los grupos sean tan pequeños?
#Desde el punto de vista práctico (Cluster para tipos de clientes), ¿tiene sentido tener muchos grupos muy pequeños? ¿Por qué?
#¿Cómo podrías resolver un problema de estos en la actualidad?...






