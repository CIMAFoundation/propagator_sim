# Crown fire: domande aperte per Nico

Questo branch (`feat/crownfire-port`) porta il modulo crown fire di
`feat/crownfire` sul `dev` attuale, sia nel core numba sia nel core Rust.
Il tuo branch non è stato toccato. Il comportamento è quello del tuo ultimo
commit (`00cdec0`); i coefficienti sono stati copiati così come sono, anche
dove abbiamo dei dubbi. Qui sotto le cose che vorremmo verificare con te prima
del merge.

Codice di riferimento: `src/propagator/core/numba/functions.py`
(`p_crowning_initiation_cruz`, `p_crowning_initiation_perrakis`,
`active_crowning_alexander`) e l'equivalente in
`rust/propagator-core/src/models.rs`. Documentazione: `docs/crown_fire.md`.

## 1. Esponente di CBD nella ROS di chioma attiva (Alexander & Cruz)

Nel tuo codice:

```python
ros_crown_active = 11.02 * U**0.9 * CBD**0.9 * exp(-0.17 * EFFM)
```

Per quanto ne so, il modello pubblicato (Cruz et al. 2005, ripreso da
Alexander & Cruz 2006) è

```
CROS = 11.02 · U10^0.90 · CBD^0.19 · exp(-0.17 · EFFM)
```

cioè l'esponente di CBD è **0.19**, non 0.9. Sembra un refuso, e la differenza
è grande. Con U = 20 km/h, CBD = 0.1 kg/m³ ed EFFM = 10 %:

| esponente | CROS (m/min) | CAC = CROS / (3/CBD) | esito |
|---|---|---|---|
| 0.9 (attuale) | 3.76 | 0.13 | passivo |
| 0.19 | 19.27 | 0.64 | passivo (ma molto più vicino alla soglia) |

Con 0.9 il fuoco di chioma attivo diventa quasi impossibile per CBD
realistici (0.05–0.3 kg/m³). Confermi 0.19?

## 2. Perrakis et al. (2023), modello 11: unità del vento e dell'umidità

Coefficienti attuali: b0 = -3.5550, b_vento = 1.4407,
b_FSG = -0.53211 (su FSG^1.5), b_vento×umidità = -0.07019,
b_SFC = 2.4897 (su ln SFC).

Con il vento in **km/h** la probabilità satura quasi subito. Esempio con
vento 20 km/h, FSG = 2 m, SFC = 1.5 kg/m², umidità 10 %:

| modello | g(x) | p |
|---|---|---|
| Cruz 2004 (km/h) | 4.79 | 0.992 |
| Perrakis, vento in km/h | 10.73 | ≈ 1.000 |
| Perrakis, vento in m/s (5.56) | 0.05 | 0.51 |

Domande:
- Nel paper il vento del modello 11 è in km/h o in m/s?
- Il termine di umidità è l'umidità del combustibile fine morto in %
  (come la usiamo noi, dalla `moisture` delle boundary conditions) oppure
  l'indice FFMC? Nel tuo codice la variabile si chiama `ffmc` ma viene
  trattata come una frazione e moltiplicata per 100, quindi le due cose
  non coincidono.

Non sono riuscito a verificare i coefficienti sul paper: se hai la tabella
sotto mano, controlla anche quelli.

## 3. Consumo di combustibile superficiale (SFC)

Sia Cruz sia Perrakis usano come SFC il carico di combustibile morto `d0`
del tipo di combustibile (kg/m²), come nel tuo codice. È voluto, o sarebbe
meglio una stima del consumo effettivo (per esempio `d0 + d1`, oppure una
frazione dipendente dall'umidità)?

Nota: con Perrakis il termine è `ln(SFC)`, quindi con `d0 = 0` il porting
restituisce p = 0 invece di `log(0)`.

## 4. Momento della decisione di crowning (differenza rispetto al tuo codice)

- **Tuo codice:** la decisione viene presa al momento del push, per ogni
  arco accettato, con il vento della cella **sorgente**, e lo stato viaggia
  sull'evento.
- **Porting:** la decisione viene presa una sola volta, quando la cella si
  accende per propagazione (`ros > 0`), con vento, umidità e chioma della
  cella **stessa**.

Lo abbiamo fatto per non aggiungere un campo all'evento nella coda e per
non cambiare il formato dei checkpoint. La probabilità per cella è la
stessa (vince comunque il primo evento che arriva), ma con vento non
uniforme il vento usato è quello della cella bersaglio. Accensioni e
atterraggi di tizzoni restano fuochi di superficie, come nel tuo codice.
Ti sembra accettabile, o c'è un motivo fisico per usare la cella sorgente?

## 5. Input dell'esempio Spotorno

- Il raster `spotorno_cbh_1m.tif` è un'altezza della base della chioma (CBH),
  ma da `ee4a857` il modello usa il fuel strata gap (FSG). Nell'esempio
  portato il CBH viene passato direttamente come FSG. Va bene, o va
  sottratta l'altezza dello strato superficiale?
- Non c'è un raster di CBD: l'esempio usa un valore uniforme (default
  0.01 kg/m³, con cui il fuoco di chioma attivo non si verifica mai). Hai
  un raster di CBD per Spotorno da aggiungere?

## 6. Citazioni da confermare

- Alexander & Cruz: nel tuo codice è citato come 2005; nella documentazione
  ho citato l'articolo su CJFR del 2006 ("Evaluating a model for predicting
  active crown fire rate of spread using wildfire observations"). Qual è
  quello giusto?
- Perrakis et al. (2023), "Improved logistic models of crown fire
  probability in Canadian conifer forests": serve la lista completa degli
  autori e il riferimento della rivista.
- Cruz et al. (2004), "Modeling the likelihood of crown fire occurrence in
  conifer forest stands": conferma che i coefficienti sono quelli del
  modello con le classi di SFC.

## Cosa non è ancora coperto

- I loader `cog` e `tiles`, la web UI e la build wasm non hanno ancora gli
  input di chioma (FSG/CBD): il crowning funziona solo in modalità `geotiff`
  e via API Python.
- Il core Rust e quello numba usano generatori casuali indipendenti, quindi
  la parità sulla decisione stocastica è statistica (come per lo spotting);
  l'esito attivo/passivo è deterministico e identico fra i due core.
