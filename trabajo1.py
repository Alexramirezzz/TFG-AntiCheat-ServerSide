import requests
import rx
from rx import operators as ops

API_KEY = "fd6ec41c"
BASE_URL = "http://www.omdbapi.com/"

def buscar_peliculas(titulo):
    params = {"apikey": API_KEY, "s": titulo}
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data.get("Search", [])

if __name__ == "__main__":
    titulo = input("Introduce un título de película para buscar: ")
    resultados = buscar_peliculas(titulo)
    print(resultados)  # <-- comprobar resultados

    rx.from_iterable(resultados).pipe(
        ops.filter(lambda item: item.get("Type", "").lower() == "movie"),
        ops.map(lambda item: f"({item.get('imdbID')}) – {item.get('Title')}: {item.get('Poster')} ({item.get('Year')})")
    ).subscribe(
        on_next=lambda msg: print(msg),
        on_error=lambda e: print("Error:", e),
        on_completed=lambda: print("\n Búsqueda completada y correcta.")
    )
