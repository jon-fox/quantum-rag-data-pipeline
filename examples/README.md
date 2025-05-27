# Examples

This directory contains example scripts demonstrating the usage of different components of the quantum energy project.

## Environment Setup

Before running the examples, ensure you have set up your environment correctly:

1.  **Create a `.env` file** in the project root directory (`/mnt/c/Developer_Workspace/quantum_work/.env`). This file should contain your API credentials:

    ```env
    ERCOT_API_USERNAME=your-ercot-username
    ERCOT_API_PASSWORD=your-ercot-password
    WEATHER_API_KEY=your-weatherapi-key
    ```

2.  **Python Path**: The example scripts automatically add the project's `src` directory to the Python path using `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))`. This allows Python to find the necessary modules.
    Alternatively, you can set the `PYTHONPATH` environment variable from your project root:
    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```

## ERCOT API Example

The `ercot_api_example.py` script demonstrates how to use the `ERCOTClient` and `ERCOTQueries` to:

*   Initialize the client and query helper with specific delivery dates.
*   Fetch 2-Day Aggregated Load Summary data.
*   Fetch 2-Day Aggregated Generation Summary data.
*   Fetch 2-Day Ancillary Service Offers.

### Running the ERCOT Example

Execute the script from the project root directory:

```bash
python -m examples.ercot_api_example
```

## Weather API Example

The `weather_api_example.py` script demonstrates how to use the `WeatherAPIClient` to:

*   Initialize the client with your API key.
*   Fetch historical hourly weather data (e.g., temperature) for multiple specified locations for a given date.
*   Combine and calculate average temperatures from the fetched data.

### Running the Weather Example

Execute the script from the project root directory:

```bash
python -m examples.weather_api_example
```
