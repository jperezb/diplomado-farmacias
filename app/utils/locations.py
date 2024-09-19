import requests
import os
import math
import json



def get_coordinates(address):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    params = {
        "address": address,
        "key": GOOGLE_API_KEY
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        results = response.json().get('results')
        if results:
            location = results[0].get('geometry').get('location')
            lat = location.get('lat')
            lng = location.get('lng')
            return lat, lng
        else:
            return None, None
    else:
        print(f"Error: {response.status_code}")
        return None, None

def calcular_distancia(coord1, coord2):
    # Fórmula de Haversine para calcular la distancia entre dos coordenadas (en kilómetros)
    R = 6371.0  # Radio de la Tierra en km

    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distancia = R * c
    return distancia

def farmacias_cercanas(user_location, data, top_n=3):
    farmacias_data = json.loads(data)

    user_lat, user_lng = user_location

    # Cálculo de distancias y selección de las farmacias más cercanas
    farmacias_con_distancia = []
    for farmacia in farmacias_data:
        farmacia_lat_str = farmacia['local_lat']
        farmacia_lng_str = farmacia['local_lng']

        # Verificar si los valores no están vacíos
        if farmacia_lat_str and farmacia_lng_str:
            try:
                farmacia_lat = float(farmacia_lat_str)
                farmacia_lng = float(farmacia_lng_str)
                distancia = calcular_distancia((user_lat, user_lng), (farmacia_lat, farmacia_lng))
                farmacias_con_distancia.append((distancia, farmacia))
            except ValueError:
                # Si ocurre un error de conversión, puedes elegir ignorar esa farmacia
                print(f"Error al convertir las coordenadas de la farmacia: {farmacia['local_nombre']}")
        else:
            print(f"Coordenadas vacías para la farmacia: {farmacia['local_nombre']}")

    # Ordenar por distancia y seleccionar las N más cercanas
    farmacias_con_distancia.sort(key=lambda x: x[0])
    farmacias_mas_cercanas = farmacias_con_distancia[:top_n]

    # Formatear el resultado
    resultado = {
        "user_location": {
            "latitud": user_lat,
            "longitud": user_lng
        },
        "farmacias": []
    }

    for distancia, farmacia in farmacias_mas_cercanas:
        resultado["farmacias"].append({
            "nombre": farmacia["local_nombre"],
            "direccion": farmacia["local_direccion"],
            "coordenadas": {
                "lat": float(farmacia["local_lat"]),
                "lng": float(farmacia["local_lng"])
            },
            "funcionamiento_hora_apertura": farmacia["funcionamiento_hora_apertura"],
            "funcionamiento_hora_cierre": farmacia["funcionamiento_hora_cierre"],
            "distancia": round(distancia, 2)  # Agregando la distancia redondeada a 2 decimales
        })

    return resultado