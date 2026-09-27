import requests

stations_url = "https://api.gios.gov.pl/pjp-api/v1/rest/station/findAll"

stations = requests.get(stations_url).json()

for stacja in stations["Lista stacji pomiarowych"]:
    if stacja.get("Nazwa miasta") == "Poznań":
        station_id = stacja["Identyfikator stacji"]

        url = f"https://api.gios.gov.pl/pjp-api/v1/rest/station/sensors/{station_id}"
        dane = requests.get(url).json()

        print("\nSTACJA:")
        print(stacja)

        for sensor in dane.get("Lista stanowisk pomiarowych", []):
            if sensor.get("Identyfikator stanowiska") == 3497:
                print("\n*** ZNALEZIONO SENSOR 3497 ***")
                print(sensor)