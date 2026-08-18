# Manuale d'uso di PROPAGATOR

## 1. A cosa serve

PROPAGATOR simula la propagazione di un incendio su una griglia raster tramite
un modello stocastico ad automi cellulari. Per ogni scenario combina:

- quota del terreno (DEM);
- tipo di combustibile presente in ogni cella;
- posizione e geometria degli inneschi;
- vento e umidità, eventualmente variabili nel tempo;
- un numero configurabile di realizzazioni probabilistiche;
- opzionalmente spotting e azioni di soppressione.

Il modo più semplice per usare il software è la CLI `propagator`. Gli esempi di
questo manuale sono pensati per PowerShell su Windows e devono essere eseguiti
dalla directory principale della repository.

## 2. Installazione

Sono richiesti Python 3.11 o successivo e `uv`.

Installare `uv`, se non è già disponibile:

```powershell
winget install --id=astral-sh.uv -e
```

Dopo l'installazione, chiudere e riaprire PowerShell e verificare:

```powershell
uv --version
```

Installare il progetto e tutte le dipendenze:

```powershell
uv sync --dev --all-extras
```

Verificare che la CLI sia disponibile:

```powershell
uv run propagator --help
```

`uv` crea e gestisce automaticamente l'ambiente virtuale `.venv`.

## 3. File di input

Per una simulazione in modalità GeoTIFF servono tre input obbligatori:

1. un file JSON con la configurazione dello scenario;
2. un GeoTIFF contenente il DEM;
3. un GeoTIFF contenente i codici dei combustibili.

La repository contiene input già pronti:

| Input | File di esempio |
| --- | --- |
| Configurazione | `example/config.json` |
| DEM | `example/dem.tif` |
| Combustibili | `example/fuel.tif` |
| Sistema di combustibili personalizzato | `example/fuel_config.yaml` |

### 3.1 DEM

Il DEM è un raster a banda singola. Il valore di ogni cella rappresenta la
quota del terreno e viene utilizzato per calcolare l'effetto della pendenza
sulla propagazione.

### 3.2 Raster dei combustibili

Il raster dei combustibili è un raster a banda singola con valori interi. Ogni
valore è l'identificativo di una classe di combustibile. Con il sistema legacy
predefinito sono presenti le seguenti classi:

| Codice | Classe |
| ---: | --- |
| 1 | Latifoglie (`broadleaves`) |
| 2 | Arbusti (`shrubs`) |
| 3 | Area non vegetata, non combustibile |
| 4 | Prateria (`grassland`) |
| 5 | Conifere (`conifers`) |
| 6 | Aree agroforestali |
| 7 | Foreste poco predisposte al fuoco |

I codici del raster devono corrispondere ai codici del sistema di combustibili
utilizzato. I codici sconosciuti vengono trattati come non vegetati.

DEM e combustibili devono avere:

- lo stesso sistema di riferimento (CRS);
- risoluzione coerente, con differenza inferiore a circa l'1%;
- estensione geografica coerente;
- griglie sovrapposte.

Il simulatore usa attualmente una dimensione interna della cella pari a 20 m.
Per risultati quantitativi attendibili, in particolare per velocità e aree, è
preferibile usare raster con celle di circa 20 m. Il campo `cellsize` è
accettato dal JSON, ma la CLI attuale non lo passa al simulatore.

### 3.3 Configurazione JSON

Un esempio minimo per una prova di 12 ore è:

```json
{
  "init_date": "202209011500",
  "ignitions": [
    "POINT: [52.51751;-6.82354]"
  ],
  "realizations": 10,
  "time_limit": 43200,
  "time_resolution": 3600,
  "do_spotting": false,
  "boundary_conditions": [
    {
      "time": 0,
      "w_dir": 0,
      "w_speed": 30,
      "moisture": 0
    },
    {
      "time": 7200,
      "w_dir": 90,
      "w_speed": 30,
      "moisture": 0
    }
  ]
}
```

Il JSON non ammette commenti. Prima di modificarlo è consigliabile crearne una
copia per ogni scenario, per esempio `config_vento_nord.json`.

#### Parametri generali

| Parametro | Obbligatorio | Significato |
| --- | --- | --- |
| `name` | No | Nome descrittivo dello scenario. |
| `init_date` | No | Data e ora UTC iniziale. Formati accettati: `YYYYMMDDHHMM`, `YYYY-MM-DDTHH:MM:SS` o `YYYY-MM-DD HH:MM:SS`. |
| `time_limit` | No | Durata massima della simulazione, in secondi. Il valore predefinito è 86400 (24 ore). |
| `time_resolution` | No | Durata di ogni avanzamento e intervallo tra gli output, in secondi. Il valore predefinito è 3600. |
| `realizations` | No | Numero di realizzazioni stocastiche. Deve essere almeno 1; valori maggiori producono probabilità più stabili ma aumentano costo e memoria. |
| `do_spotting` | No | Se `true`, abilita il trasporto di tizzoni per i combustibili predisposti. |
| `ros_model` | No | Modello della velocità di propagazione: `wang` (predefinito) o `rothermel`. |
| `prob_moist_model` | No | Modello dell'effetto dell'umidità: `trucchia` (predefinito) o `baghino`. |
| `epsg` | No | CRS dichiarato per le geometrie. Per la CLI attuale usare EPSG:4326. |
| `ignitions` | Condizionale | Inneschi iniziali. Possono essere qui oppure nella condizione al tempo 0. |
| `boundary_conditions` | Sì | Elenco non vuoto delle condizioni applicate ai tempi indicati. Deve contenere il tempo 0. |

Conversioni temporali utili:

| Durata | Secondi |
| ---: | ---: |
| 1 ora | 3600 |
| 6 ore | 21600 |
| 12 ore | 43200 |
| 24 ore | 86400 |
| 40 ore | 144000 |

#### Condizioni al contorno

Ogni elemento di `boundary_conditions` può contenere:

| Parametro | Significato |
| --- | --- |
| `time` | Secondi trascorsi dall'inizio; deve essere maggiore o uguale a 0. Non sono ammessi tempi duplicati. |
| `w_dir` | Direzione del vento in gradi, in senso orario a partire dal nord: 0° nord, 90° est, 180° sud, 270° ovest. |
| `w_speed` | Velocità del vento in km/h. |
| `moisture` | Umidità del combustibile in percentuale, tra 0 e 100. |
| `ignitions` | Nuovi inneschi applicati in quel momento. |
| `actions` | Eventuali azioni di soppressione. |

Deve essere presente una condizione con `time: 0`. Dopo l'unione con il campo
`ignitions` generale, questa condizione deve contenere almeno un innesco. I
parametri omessi in una condizione successiva non introducono un nuovo campo e
lasciano valido lo stato già impostato.

Nell'esempio precedente il vento soffia inizialmente con direzione 0° e, dopo
7200 secondi (2 ore), cambia a 90°.

#### Formato delle geometrie

Le geometrie della CLI legacy sono espresse normalmente in WGS84. Il formato
usa latitudine prima della longitudine:

```text
POINT: [latitudine;longitudine]
LINE:[lat1 lat2 lat3];[lon1 lon2 lon3]
POLYGON:[lat1 lat2 lat3 lat1];[lon1 lon2 lon3 lon1]
```

Esempi:

```json
"ignitions": [
  "POINT: [52.51751;-6.82354]"
]
```

```json
"ignitions": [
  "LINE:[52.51 52.52];[-6.82 -6.81]"
]
```

Il punto o la geometria devono ricadere nell'estensione dei raster.

### 3.4 Sistema di combustibili personalizzato

Senza `--fuel-config` viene usato il sistema legacy. Per usare classi proprie,
creare un YAML con un nodo principale `fuels`. Il file
`example/fuel_config.yaml` è il riferimento completo.

Per ogni combustibile si possono impostare:

| Parametro | Significato |
| --- | --- |
| `name` | Nome della classe. |
| `v0` | Velocità nominale di propagazione in m/h; viene convertita internamente in m/min. |
| `d0` | Carico di combustibile morto in kg/m². |
| `d1` | Carico di combustibile vivo in kg/m², opzionale. |
| `hhv` | Potere calorifico superiore in kJ/kg. |
| `humidity` | Umidità del combustibile vivo in percentuale, opzionale. |
| `spread_probability` | Probabilità di propagazione verso ciascuna classe, tra 0 e 1. |
| `spotting` | Indica se la classe può generare tizzoni. |
| `prob_ign_by_embers` | Probabilità di accensione dovuta ai tizzoni, tra 0 e 1. |
| `burn` | Indica se la classe è combustibile. Deve esistere una classe con `burn: false`. |

## 4. Comandi per avviare le simulazioni

### 4.1 Simulazione base con i dati di esempio

```powershell
$env:PROJ_LIB = "$PWD\.venv\Lib\site-packages\pyproj\proj_dir\share\proj"

uv run propagator --config example/config.json --mode geotiff --dem example/dem.tif --fuel example/fuel.tif --output results/test-12h --verbose
```

### 4.2 Simulazione con sistema di combustibili personalizzato

```powershell
uv run propagator --config example/config.json --mode geotiff --dem example/dem.tif --fuel example/fuel.tif --fuel-config example/fuel_config.yaml --output results/test-custom-fuels --verbose
```

### 4.3 Simulazione con registrazione del log

```powershell
uv run propagator --config example/config.json --mode geotiff --dem example/dem.tif --fuel example/fuel.tif --output results/test-con-log --verbose --record
```

Questa modalità crea anche `run.log` e `run.html` nella cartella dei risultati.

### 4.4 Continuare quando il fuoco raggiunge il bordo

```powershell
uv run propagator --config example/config.json --mode geotiff --dem example/dem.tif --fuel example/fuel.tif --output results/test-bordi --verbose --ignore-out-of-bounds
```

Senza `--ignore-out-of-bounds` la simulazione si interrompe quando almeno un
fronte raggiunge il bordo del dominio. L'opzione permette di continuare, ma il
risultato vicino al bordo rappresenta un incendio tagliato dall'estensione del
raster. Per studi reali è preferibile ampliare il dominio.

### 4.5 Opzioni principali della CLI

| Opzione | Significato |
| --- | --- |
| `--config PATH` | Configurazione JSON. |
| `--mode geotiff` | Caricamento da due GeoTIFF espliciti. |
| `--dem PATH` | Percorso del DEM. |
| `--fuel PATH` | Percorso del raster dei combustibili. |
| `--fuel-config PATH` | Sistema di combustibili YAML opzionale. |
| `--output PATH` | Directory di destinazione, creata automaticamente. |
| `--isochrones ...` | Soglie probabilistiche delle isocrone; valori predefiniti 0.5, 0.75 e 0.9. |
| `--verbose` | Mostra configurazione, condizioni e avanzamento. |
| `--record` | Salva il log della console. |
| `--ignore-out-of-bounds` | Non interrompe la simulazione al bordo della griglia. |

Usare una directory di output diversa per ogni scenario. Riutilizzare la stessa
directory può lasciare insieme file appartenenti a esecuzioni differenti.

## 5. Dove trovare gli output

Gli output si trovano nella directory indicata con `--output`. Per esempio:

```text
results/test-12h/
```

Il suffisso numerico del nome è il tempo simulato in secondi. Per esempio:

```text
fire_probability_43200.tiff
```

è la probabilità di incendio dopo 43200 secondi, cioè 12 ore dall'istante
`init_date`.

Con la CLI attuale può essere scritto uno snapshot successivo a `time_limit`,
perché il limite viene verificato dopo l'avanzamento. Per un'analisi a 12 ore
usare i file con suffisso `_43200`, anche se è presente `_46800`.

## 6. Come interpretare gli output

### 6.1 Raster GeoTIFF

I GeoTIFF sono a banda singola e vengono esportati in WGS84 (EPSG:4326). Possono
essere aperti in QGIS, ArcGIS o Python con `rasterio`.

| File | Unità | Interpretazione |
| --- | --- | --- |
| `fire_probability_T.tiff` | 0–1 | Frazione delle realizzazioni nelle quali la cella è bruciata entro il tempo `T`. |
| `min_arrival_time_T.tiff` | secondi | Primo tempo di arrivo osservato tra le realizzazioni in cui la cella è bruciata. |
| `mean_arrival_time_T.tiff` | secondi | Tempo medio di arrivo calcolato sulle realizzazioni in cui la cella è bruciata. |
| `ros_mean_T.tiff` | m/min | Velocità media di propagazione nelle realizzazioni in cui la cella è bruciata. |
| `ros_max_T.tiff` | m/min | Massima velocità di propagazione osservata. |
| `fireline_intensity_mean_T.tiff` | kW/m | Intensità lineare media del fronte. |
| `fireline_intensity_max_T.tiff` | kW/m | Massima intensità lineare osservata. |
| `spotting_generation_probability_T.tiff` | 0–1 | Probabilità che la cella abbia generato almeno un tizzone; presente con spotting attivo. |
| `spotting_receiving_probability_T.tiff` | 0–1 | Probabilità che la cella sia stata raggiunta da un tizzone; presente con spotting attivo. |

Esempio di interpretazione di `fire_probability` con 10 realizzazioni:

- valore `0`: la cella non è bruciata in nessuna realizzazione;
- valore `0.3`: la cella è bruciata in 3 realizzazioni su 10;
- valore `1`: la cella è bruciata in tutte le realizzazioni.

Con sole 10 realizzazioni la probabilità cambia a intervalli di 0.1. Per analisi
più stabili aumentare `realizations`, tenendo conto dell'aumento dei tempi di
calcolo.

I tempi di arrivo devono essere letti insieme a `fire_probability`: una cella
può avere un tempo minimo molto precoce ma una probabilità di incendio molto
bassa. Il valore 0 nei raster temporali indica normalmente una cella mai
raggiunta, oltre alle celle accese all'istante iniziale; non va interpretato da
solo come un tempo di arrivo certo.

### 6.2 Isocrone GeoJSON

Il file `isochrones_T.json` contiene linee corrispondenti alle soglie di
probabilità richieste, normalmente 0.5, 0.75 e 0.9. Una linea a soglia 0.75
delimita la zona ricavata dalle celle con probabilità almeno pari al 75%.

Le isocrone sono utili per una visualizzazione sintetica, ma derivano da
filtraggio e regolarizzazione del raster. Per analisi numeriche usare il
GeoTIFF `fire_probability_T.tiff`.

### 6.3 Metadati JSON

Il file `metadata_T.json` contiene:

| Campo | Significato |
| --- | --- |
| `c_time` | Tempo simulato in secondi. |
| `ref_date` | Data e ora UTC corrispondenti allo snapshot. |
| `n_active` | Numero di realizzazioni con un fronte ancora attivo. |
| `area_mean` | Area attesa bruciata, in m², ottenuta sommando le probabilità delle celle. |
| `area_50` | Area in m² con probabilità almeno 0.50. |
| `area_75` | Area in m² con probabilità almeno 0.75. |
| `area_90` | Area in m² con probabilità almeno 0.90. |

Per convertire le aree da m² a ettari dividere per 10000; per convertirle in
km² dividere per 1000000.

### 6.4 Log

Se la simulazione è avviata con `--record`, nella directory di output sono
presenti:

- `run.log`, leggibile come testo;
- `run.html`, versione formattata del log.

Il log è il primo riferimento per controllare arresti anticipati, errori nei
dati e raggiungimento dei bordi.

## 7. Procedura consigliata per una prova

1. Copiare `example/config.json` assegnando un nome riconoscibile allo
   scenario.
2. Impostare `time_limit`, `time_resolution`, `realizations`, innesco, vento e
   umidità.
3. Controllare che l'innesco ricada dentro i raster.
4. Scegliere una nuova directory sotto `results/`.
5. Avviare la simulazione con `--verbose --record`.
6. Verificare che esista lo snapshot corrispondente al tempo desiderato.
7. Aprire `fire_probability_T.tiff` in QGIS e sovrapporlo a DEM e combustibili.
8. Consultare `metadata_T.json` per le aree e `run.log` per eventuali avvisi.

## 8. Problemi comuni

### `uv` non riconosciuto

Installare `uv` con `winget`, quindi chiudere e riaprire PowerShell.

### Configurazione rifiutata

Controllare che:

- `boundary_conditions` non sia vuoto;
- esista una condizione con `time: 0`;
- sia presente almeno un innesco iniziale;
- non esistano condizioni con lo stesso `time`;
- `moisture` sia compreso tra 0 e 100;
- il JSON non contenga commenti o virgole finali.

### Simulazione terminata prima del limite

Il fuoco ha probabilmente raggiunto il bordo del raster. Controllare il log.
Per una prova si può usare `--ignore-out-of-bounds`; per un'analisi attendibile
è meglio usare un dominio più esteso.

### Risultati mescolati

Non riutilizzare la directory di una simulazione precedente. Creare una nuova
cartella di output per ogni scenario.

## 9. Altra documentazione

- [Guida introduttiva](docs/getting-started.md)
- [Riferimento della CLI](docs/cli.md)
- [Descrizione degli output](docs/outputs.md)
- [Uso programmatico](docs/programmatic.md)
- [Spotting](docs/spotting.md)

