import python_weather


async def temp_callback() -> str:
    async with python_weather.Client(unit=python_weather.IMPERIAL) as weather_client:  # type: ignore
        current_weather = await weather_client.get("Fayetteville, AR")
        temperature = current_weather.temperature
        return f"The current temperature in Fayetteville, AR is {temperature}°F."
