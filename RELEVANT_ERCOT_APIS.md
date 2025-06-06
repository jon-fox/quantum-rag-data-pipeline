1. 2-Day Aggregated Load Summary
Why: Core metric for daily demand.

Use: Report total and peak load, compare against previous day or forecast.

2. 2-Day Aggregated Generation Summary
Why: Supply-side total; check if generation met load.

Use: Detect generation shortfalls or excess.

3. 2-Day Aggregated Ancillary Service Offers ECRSS
Why: Indicates stress or grid contingency planning.

Use: Flag elevated offers as signs of tight reserve margins.

4. 2-Day Aggregated DSR Loads
Why: Shows demand-side response activity.

Use: Suggests ERCOT called for load reduction due to grid stress.

5. 2-Day Aggregated Output Schedule
Why: Planned vs. actual output indicator.

Use: Spot mismatches between scheduled and actual generation.

---

### 🌤️ Weather Data Collection Strategy

To support statewide ERCOT load and generation forecasting, weather data is collected from major metropolitan areas representing each geographic region of the ERCOT grid. This ensures broad spatial coverage while minimizing data redundancy.

**Selected Cities by Region:**

* **North** – Dallas, TX
* **South** – San Antonio, TX
* **Central** – Austin, TX
* **Coastal / Southeast** – Houston, TX
* **West** – El Paso, TX or Lubbock, TX
* *(Optional)* **Gulf Coast** – Corpus Christi, TX

**Collected Variables (Hourly):**

* Temperature at 2 meters (`temperature_2m`)
* Relative humidity (`relative_humidity_2m`) *(optional but useful)*
* Precipitation (`precipitation`) *(optional if available)*

Data is pulled daily from a historical weather API (e.g., WeatherAPI or Open-Meteo) and averaged across locations to approximate overall ERCOT weather trends.

This data is then used to:

* Contextualize load/generation behavior
* Explain grid stress or reserve changes
* Feed into semantic summaries in the RAG pipeline

---

**Sample Summary (Statewide ERCOT):**
ISO: ERCOT
Date_from: 2025-05-08
Date_to:   2025-05-09
Avg system load: 51405 MW
Telemetry generation: 51438 MW
DAM HubAvg price: 32.53 $/MWh
Renewables: 16993 MW (wind 5490 MW | solar 7655 MW | other 3847 MW) (33%)
ECRSS max offer: 4404 MW
DSR load: 219 MW
SCED dispatchable: 4270 MW (headroom LSL 2997 MW | HSL 16380 MW)
Base-point non-intermittent: 34502 MW (SH 41011 MW | SL 19636 MW)
Avg Texas temp: 21.9 °C


<!-- 
May include later
**Weather Impact:** No major weather anomalies were recorded, supporting typical late spring demand trends. 
-->


---

**Notes on Summary Design:**

* Quantities are rounded and normalized (e.g., GWh for energy, °F for temperature).
* Summary focuses on system-wide energy balance, consumption, generation, and operational flexibility.
* Weather coverage is concise, highlighting only relevant temperature ranges across key regions.
* Summary avoids excessive detail and noise while retaining key metrics useful for forecasting.
* Easily extendable to include historical comparisons or trends.