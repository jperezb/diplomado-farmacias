# Usa una imagen base oficial de Python
# docker run -p 8900:8000 mi-app-fastapi00
FROM python:3.12-slim

# Establece el entorno para que no requiera entradas interactivas durante la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# Establece la zona horaria (cambia "America/Santiago" por la que desees)
ENV TZ=America/Santiago

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de la aplicación
COPY . .

# Actualiza el gestor de paquetes e instala 'tzdata' para la configuración de la zona horaria
RUN apt-get update -y && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir --no-deps -r requirements.txt

# Exponer el puerto por defecto de Uvicorn
EXPOSE 8000

# Comando para ejecutar la aplicación
#CMD ["uvicorn", "appfastapi:app", "--host", "0.0.0.0", "--port", "8900", "--log-level", "debug"]
CMD ["uvicorn", "appfastapi:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]

#uvicorn appfastapi:app --host 0.0.0.0 --port 8900 --log-level debug
