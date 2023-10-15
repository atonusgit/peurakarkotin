def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = f.read().strip()
        return f"{int(temp) // 1000}°C"
    except FileNotFoundError:
        return "N/A"

print(get_cpu_temp())
