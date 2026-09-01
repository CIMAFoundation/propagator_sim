# PROPAGATOR — Manuale utente

## Indice

1. [Cos'è PROPAGATOR](#1-cosè-propagator)
2. [Come si comporta un incendio boschivo](#2-come-si-comporta-un-incendio-boschivo)
3. [Guida all'uso della web app](#3-guida-alluso-della-web-app)
4. [Le variabili di input, spiegate](#4-le-variabili-di-input-spiegate)
5. [Come funziona l'algoritmo (in termini semplici)](#5-come-funziona-lalgoritmo-in-termini-semplici)
6. [Gli output e come leggerli](#6-gli-output-e-come-leggerli)
7. [Interventi antincendio](#7-interventi-antincendio)
8. [Dati geografici: DEM, copertura del suolo e caching](#8-dati-geografici-dem-copertura-del-suolo-e-caching)
9. [Punti di interesse (POI) da OpenStreetMap](#9-punti-di-interesse-poi-da-openstreetmap)
10. [Usare l'API direttamente](#10-usare-lapi-direttamente)
11. [Limiti del modello](#11-limiti-del-modello)

## 1. Cos'è PROPAGATOR

PROPAGATOR è un simulatore di propagazione di incendi boschivi sviluppato
dalla Fondazione CIMA Research. Dato un punto di innesco, uno scenario
meteorologico e un'area geografica reale (rilievo del terreno e copertura
vegetale), PROPAGATOR stima come un incendio potrebbe propagarsi nelle ore
successive: dove è più probabile che il fuoco arrivi, con quale velocità e
con quale intensità.

Vale la pena chiarire subito una cosa: il fuoco non si comporta mai in modo
perfettamente prevedibile, perché dipende da molti fattori piccoli e in
parte casuali (una raffica di vento, una chiazza di combustibile più secco,
un tizzone trasportato più lontano del previsto). Per questo PROPAGATOR non
produce un'unica previsione fissa, ma una **probabilità**: esegue lo stesso
scenario molte volte, lasciando che il caso giochi un ruolo leggermente
diverso ogni volta, e mostra poi quante volte il fuoco ha raggiunto ogni
punto della mappa. È uno strumento di supporto alle decisioni — utile per
capire dove concentrare attenzione e risorse — non una previsione esatta.

## 2. Come si comporta un incendio boschivo

Perché un incendio si propaga nel modo in cui è noto propagarsi? Quattro
fattori principali governano il comportamento del fuoco, e sono esattamente
quelli che l'app chiede di impostare prima di avviare una simulazione.

### Vento

Il vento spinge fiamme e calore verso il combustibile non ancora bruciato
davanti al fronte, preriscaldandolo e facendolo incendiare più rapidamente:
il fronte del fuoco avanza molto più velocemente *sottovento* (nella
direzione verso cui soffia il vento) rispetto alle altre direzioni. Più
forte è il vento, più questo effetto si accentua, e più l'incendio assume
una forma allungata, "a lingua", nella direzione del vento.

### Pendenza del terreno

Il fuoco sale più velocemente di quanto scenda. In salita, le fiamme si
trovano più vicine al combustibile sovrastante (preriscaldandolo ancor
prima di raggiungerlo), mentre in discesa le fiamme si allontanano dal
pendio e il calore si disperde maggiormente. Una pendenza ripida può quindi
far correre un incendio molto più velocemente verso una cresta che verso
una valle.

### Umidità del combustibile

Legno, erba e arbusti umidi sono più difficili da bruciare: parte del
calore serve a far evaporare l'acqua contenuta nel combustibile invece di
alimentare la combustione. Il combustibile secco (dopo giorni senza
pioggia, con bassa umidità) si incendia più facilmente e brucia più
velocemente; il combustibile umido rallenta o può addirittura fermare
l'avanzata del fronte. Bagnare il combustibile davanti all'incendio (ad
esempio con acqua sganciata da aeromobili) è esattamente una delle tattiche
antincendio che si possono simulare in PROPAGATOR — vedi la sezione sugli
interventi antincendio.

### Tipo di vegetazione

Non tutta la vegetazione brucia allo stesso modo. Una foresta di conifere
(pino, abete) contiene resine altamente infiammabili e brucia con fiamme
alte e intense; un prato si incendia facilmente ma brucia rapidamente e con
minore intensità rispetto a una foresta; un terreno agricolo o non vegetato
può non bruciare affatto, o può rallentare notevolmente l'incendio, agendo
quasi come una barriera naturale. PROPAGATOR conosce diverse categorie di
vegetazione (foreste di latifoglie, foreste di conifere, arbusteti, terreni
agricoli, prati, aree non vegetate...) e assegna a ciascuna un
comportamento diverso in termini di velocità e probabilità di propagazione.

### Tizzoni trasportati dal vento (spotting)

Negli incendi più intensi, specialmente nelle foreste di conifere, il vento
può sollevare frammenti in combustione (tizzoni, aghi di pino accesi,
piccoli rametti) e trasportarli centinaia di metri davanti al fronte
principale. Se un tizzone atterra su combustibile secco può innescare un
nuovo focolaio isolato, che poi cresce autonomamente e può ricongiungersi
al fronte principale o addirittura superarlo. Questo fenomeno si chiama
**spotting** ed è uno dei motivi per cui gli incendi boschivi possono
"saltare" ostacoli come strade o corsi d'acqua che altrimenti li
fermerebbero. PROPAGATOR può opzionalmente simulare anche questo effetto.

## 3. Guida all'uso della web app

L'interfaccia è divisa in un pannello di controllo a sinistra e una mappa a
destra. I passaggi per eseguire una simulazione sono:

1. **Scegli il centro** — clicca il pulsante e poi clicca sulla mappa, nel
   punto intorno al quale vuoi analizzare l'area (tipicamente il paese o la
   zona a rischio). Poi regola il **Raggio** (quanto estesa è l'area da
   scaricare e simulare, fino a 50 km) e la **Dimensione cella** (la
   risoluzione della griglia).
2. **Scegli l'innesco** — clicca il pulsante e poi clicca sulla mappa nel
   punto in cui l'incendio dovrebbe iniziare. Deve trovarsi su terreno
   vegetato (foresta, arbusteto, prato...), non su un'area edificata o su
   terreno non infiammabile: in tal caso l'app mostra un avviso.
3. **Meteo** — imposta **Direzione del vento**, **Velocità del vento** e
   **Umidità** del combustibile, per lo scenario che vuoi analizzare.
4. **Simulazione** — imposta quante ore simulare (**Durata**), ogni quanto
   salvare un risultato (**Risoluzione temporale**), quante volte ripetere
   la simulazione (**Realizzazioni**), se includere i tizzoni
   (**Spotting**) e le soglie di probabilità disegnate come linee sulla
   mappa (**Soglie isocrone**).
5. **Interventi antincendio** (opzionale) — se vuoi simulare un intervento
   antincendio attivo, scegli il **Tipo** di mezzo, quando avviene (**Tempo
   dell'azione**), poi clicca **Disegna linea**, traccia il percorso sulla
   mappa e premi **Termina linea**. Puoi mettere in coda più interventi e
   rimuoverli dalla lista.
6. **Avvia simulazione** — il pulsante diventa attivo una volta impostati
   sia il centro che l'innesco. Durante l'esecuzione della simulazione, una
   barra di avanzamento mostra la fase corrente (scaricamento dati,
   esecuzione...).
7. Al termine, la mappa mostra l'incendio simulato e appare un pannello con
   le statistiche, un grafico di crescita dell'area e uno slider **Tempo**
   per scorrere i risultati ora per ora. Vedi la sezione sugli output per
   come interpretarli.

## 4. Le variabili di input, spiegate

**Raggio**
: Il raggio, in km, dell'area intorno al centro scelto che verrà scaricata
  e simulata. Un'area più estesa copre più territorio ma richiede più
  tempo di calcolo e più dati da scaricare.

**Dimensione cella**
: La dimensione, in metri, di ogni cella della griglia usata per la
  simulazione: è il "pixel" con cui viene rappresentato il terreno. Valori
  più piccoli (ad es. 20 m) danno una mappa più dettagliata ma richiedono
  più tempo di calcolo; valori più grandi (ad es. 100 m) sono più veloci
  ma meno precisi.

**Direzione del vento**
: La direzione da cui soffia il vento, in gradi dal nord (0°=Nord,
  90°=Est, 180°=Sud, 270°=Ovest). L'incendio tenderà ad avanzare più
  velocemente nella direzione opposta, ossia la direzione verso cui
  soffia il vento.

**Velocità del vento**
: Intensità del vento, in km/h. Più è alta, più velocemente avanza il
  fronte sottovento, e più allungata diventa la forma dell'incendio.

**Umidità**
: Umidità del combustibile, in percentuale. Valori alti rappresentano
  combustibile più umido (ad es. dopo la pioggia) e rallentano la
  propagazione; valori bassi rappresentano combustibile secco e la
  accelerano.

**Durata**
: Quante ore simulare l'evoluzione dell'incendio a partire dall'innesco.

**Risoluzione temporale**
: Ogni quante ore viene salvato un "fotogramma" dei risultati (mappa e
  statistiche), da scorrere in seguito con lo slider **Tempo**. Non
  influisce sulla precisione della simulazione, solo sulla frequenza con
  cui vedi un aggiornamento.

**Realizzazioni**
: Quante volte viene simulato lo stesso scenario, lasciando che gli
  elementi casuali del modello (vedi la sezione successiva) diano un
  esito leggermente diverso ogni volta. Il risultato finale è la frazione
  di realizzazioni in cui l'incendio ha raggiunto ogni punto, ossia una
  probabilità. Più realizzazioni danno una stima più affidabile ma
  richiedono più tempo di calcolo.

**Spotting (tizzoni)**
: Se abilitato, il modello genera tizzoni trasportati dal vento (vedi
  sezione 2) che possono innescare nuovi focolai davanti al fronte
  principale, in particolare nelle foreste di conifere.

**Soglie isocrone**
: Una o più soglie di probabilità (tra 0 e 1) per cui viene disegnata una
  linea di contorno (isocrona) sulla mappa: ad esempio "0.5" disegna il
  confine dell'area in cui l'incendio è arrivato in almeno metà delle
  realizzazioni.

## 5. Come funziona l'algoritmo (in termini semplici)

PROPAGATOR rappresenta il terreno come una griglia di celle (la dimensione
della cella è la **Dimensione cella** scelta). All'inizio, solo la cella (o
le celle) di innesco è "in fiamme". Da lì, l'incendio può propagarsi a ogni
cella vicina: per ogni coppia di celle vicine, il modello calcola due cose:

- **Quanto è probabile** che l'incendio si propaghi da una cella all'altra;
- **Quanto tempo impiega**, cioè quanto velocemente si propaga (la
  cosiddetta velocità di propagazione, o ROS).

Entrambi i valori partono da un comportamento base che dipende dal **tipo
di vegetazione** delle due celle coinvolte (ogni categoria di vegetazione
ha una propria velocità e probabilità di propagazione tipiche, calibrate su
osservazioni reali), e vengono poi corretti in base al **vento**, alla
**pendenza** tra le due celle e all'**umidità del combustibile**,
esattamente come descritto nella sezione 2: propagarsi verso una cella
sottovento o in salita diventa più probabile e più veloce, propagarsi verso
una cella sopravento, in discesa o più umida diventa meno probabile e più
lento.

Questo processo si ripete cella per cella, seguendo il fronte dell'incendio
mentre avanza nel tempo: questo è ciò che si chiama *automa cellulare*.
Poiché probabilità e casualità intervengono a ogni passo, una singola
simulazione è solo uno dei tanti modi in cui l'incendio potrebbe evolvere.
Per questo PROPAGATOR ripete l'intera simulazione tante volte quante sono
le **Realizzazioni** scelte (una tecnica chiamata simulazione Monte Carlo),
e conta in quante di queste ripetizioni l'incendio ha raggiunto ogni cella.
Quel conteggio, diviso per il numero di realizzazioni, è la mappa di
probabilità che vedi come risultato finale.

Se lo **spotting** è abilitato, a ogni passo le celle in fiamme di tipi di
vegetazione "soggetti a spotting" (tipicamente le conifere) possono anche
generare tizzoni che volano nella direzione del vento e, se atterrano su
combustibile infiammabile, innescano un nuovo focolaio piccolo e
indipendente, che poi si propaga secondo le stesse regole del fronte
principale.

## 6. Gli output e come leggerli

**Mappa di probabilità**
: La colorazione sulla mappa rappresenta, per ogni cella, in quante delle
  realizzazioni simulate l'incendio ha raggiunto quel punto: più intenso è
  il colore, più è probabile che l'incendio raggiunga quell'area, dato lo
  scenario impostato.

**Isocrone**
: Le linee disegnate sulla mappa in corrispondenza delle **Soglie
  isocrone** scelte: delimitano l'area in cui la probabilità di incendio è
  almeno pari alla soglia. Utili per individuare rapidamente, ad esempio,
  "l'area ad alto rischio" (soglia 0.9) rispetto a "l'area a possibile
  rischio" (soglia 0.5).

**Slider Tempo / fotogramma**
: Scorrendo lo slider si vede come cambiano mappa e statistiche ora per
  ora, negli istanti salvati in base alla **Risoluzione temporale** scelta:
  questo permette di capire non solo dove, ma anche quando l'incendio
  potrebbe raggiungere un dato punto.

**Area media (ha)**
: L'area bruciata attesa, in ettari, calcolata sommando la probabilità di
  incendio di ogni cella. È una stima "media" che tiene conto
  dell'incertezza: non l'area di un singolo scenario, ma l'area attesa su
  tutte le realizzazioni.

**Area prob≥50%**
: L'area, in ettari, in cui l'incendio ha raggiunto almeno metà delle
  realizzazioni simulate: una stima più "prudenziale" e concreta di dove
  l'incendio è più probabile che sia effettivamente passato.

**Realizzazioni attive**
: Quante delle simulazioni ripetute hanno ancora un fronte di fuoco in
  movimento in quell'istante (le altre si sono già fermate, ad esempio
  perché l'incendio ha esaurito il combustibile raggiungibile).

**Fotogrammi disponibili**
: Quanti istanti temporali sono stati salvati e sono disponibili da
  scorrere con lo slider.

**Grafico di crescita dell'area**
: Mostra come l'area media attesa (linea continua) e l'area con
  probabilità almeno del 50% (linea tratteggiata) crescono nel tempo:
  utile per farsi un'idea di quanto velocemente si espande l'incendio
  nello scenario impostato.

## 7. Interventi antincendio

PROPAGATOR permette di simulare l'effetto di un intervento antincendio
attivo, disegnando una linea sulla mappa (il percorso del mezzo, o dove
avviene l'intervento) e scegliendo quando avviene, rispetto all'inizio
della simulazione:

**Canadair**
: Simula uno sgancio d'acqua aereo: aumenta l'umidità del combustibile
  lungo la linea disegnata (e in un buffer di sicurezza intorno ad essa),
  rendendo più difficile per l'incendio proseguire in quell'area.
  L'effetto non è permanente: l'umidità aggiunta decade nel tempo, circa
  l'1% al minuto, come l'acqua che evapora o si infiltra nel terreno.

**Elicottero**
: Simile allo sgancio del Canadair ma con sganci puntuali più sparsi lungo
  la linea, con un effetto leggermente minore sull'umidità.

**Linea d'acqua**
: Simula un intervento a terra con manichette (una linea d'acqua
  continua): aumenta l'umidità lungo l'intera linea disegnata e il suo
  buffer circostante, con l'effetto più forte tra gli interventi a base
  d'acqua disponibili.

**Mezzi pesanti**
: Simula la creazione di una fascia tagliafuoco (ad esempio con un
  bulldozer): non bagna il combustibile, ma lo rimuove fisicamente lungo
  la linea disegnata, rendendo quella fascia permanentemente non
  infiammabile per il resto della simulazione.

## 8. Dati geografici: DEM, copertura del suolo e caching

Per eseguire una simulazione su un'area reale, PROPAGATOR ha bisogno di due
mappe raster allineate sulla stessa griglia: il **rilievo del terreno**
(DEM, Digital Elevation Model, da cui vengono calcolate le pendenze) e la
**copertura del suolo/vegetazione** (da cui viene derivato il tipo di
combustibile di ogni cella). Entrambe vengono scaricate automaticamente,
al bisogno, da fonti pubbliche gratuite, senza necessità di chiave API.

### Da dove vengono scaricati i dati

**Rilievo del terreno — Copernicus DEM GLO-30**
: Un modello digitale del terreno globale a risoluzione di circa 30 m,
  prodotto dall'Agenzia Spaziale Europea (ESA) nell'ambito del programma
  Copernicus, distribuito come Cloud-Optimized GeoTIFF (COG) pubblici su
  Amazon S3 (`copernicus-dem-30m.s3.amazonaws.com`). I dati sono
  organizzati in tile di 1°&times;1° di lato; l'app scarica solo le tile
  che coprono l'area richiesta (in base al centro e al **Raggio**
  scelti).

**Copertura del suolo — ESA WorldCover 10 m (v200, 2021)**
: Una mappa globale della copertura del suolo a risoluzione di 10 m,
  anch'essa prodotta dall'ESA a partire da immagini satellitari
  Sentinel-1/2, distribuita come COG pubblici su Amazon S3
  (`esa-worldcover.s3.eu-central-1.amazonaws.com`). Le tile sono di
  3°&times;3°. WorldCover classifica il territorio in categorie generiche
  (copertura arborea, arbusteti, prati, coltivi, aree urbanizzate, corpi
  d'acqua, ecc.), che PROPAGATOR poi traduce nelle proprie categorie di
  combustibile (vedi sotto).

Entrambe le fonti sono aggiornate periodicamente dai rispettivi enti, ma
PROPAGATOR usa sempre le versioni fisse indicate sopra (GLO-30 per il DEM,
v200/2021 per WorldCover): non effettua un controllo automatico di nuove
versioni.

### Cosa rappresentano i dati e come diventano input della simulazione

Per l'area richiesta (un quadrato centrato sul punto scelto, di lato pari
al doppio del **Raggio**), l'app:

1. individua e scarica le tile DEM e WorldCover che coprono quel quadrato;
2. le riproietta e le "mosaica" (le unisce) su un'unica griglia allineata,
   nella proiezione UTM locale, con celle della **Dimensione cella**
   scelta (il DEM tramite interpolazione bilineare, la copertura del
   suolo tramite campionamento del vicino più prossimo, per non mescolare
   categorie diverse);
3. converte ogni classe WorldCover nella corrispondente categoria di
   combustibile di PROPAGATOR, secondo questa mappatura:

   | Classe WorldCover           | Codice | Categoria PROPAGATOR         |
   | ---------------------------- | ------ | ----------------------------- |
   | Copertura arborea (10)        | 1      | Latifoglie                    |
   | Arbusteti (20)                | 2      | Arbusteti                     |
   | Prati (30)                    | 4      | Prati                         |
   | Coltivi (40)                  | 6      | Terreni agro-forestali         |
   | Aree edificate (50)           | 3      | Non vegetato                  |
   | Aree nude/sparsa vegetazione (60) | 3  | Non vegetato                  |
   | Neve e ghiaccio (70)          | 3      | Non vegetato                  |
   | Corpi d'acqua permanenti (80) | 3      | Non vegetato                  |
   | Zone umide erbacee (90)       | 3      | Non vegetato                  |
   | Mangrovie (95)                | 1      | Latifoglie (raro fuori tropici) |
   | Muschi e licheni (100)        | 3      | Non vegetato                  |

   WorldCover non distingue tra foreste di latifoglie e di conifere: la
   classe "copertura arborea" viene quindi assegnata di default alle
   latifoglie, una scelta ragionevole per il centro Italia ma
   approssimativa altrove. Questa mappatura automatica è pensata per
   ottenere rapidamente una simulazione plausibile, non sostituisce una
   mappa del combustibile costruita ad hoc (ad es. da Corine Land Cover
   incrociato con dati sulle specie forestali presenti).

Se un **punto di innesco** ricade su una cella di categoria "non vegetato",
l'app lo segnala con un avviso, perché l'incendio si spegnerebbe subito.

### Meccanismo di caching

Scaricare le stesse tile ogni volta sarebbe lento e inutile, dato che le
fonti dati non cambiano tra una simulazione e l'altra sulla stessa zona.
Per questo ogni tile scaricata (DEM o WorldCover) viene salvata su disco
in una cartella di cache locale e riutilizzata alle richieste successive:

- la cache di default si trova in `~/.propagator/cache/` (sotto-cartelle
  `dem/` e `worldcover/`), ma l'app web può usare una cartella diversa a
  seconda della configurazione del server;
- una tile viene scaricata solo se il file corrispondente non esiste
  ancora nella cache: se è già presente, viene riusata direttamente,
  senza alcuna richiesta di rete;
- il download di ogni file avviene prima in un file temporaneo
  (`.part`), rinominato nel nome definitivo solo a scaricamento
  completato: un'interruzione a metà download (rete caduta,
  processo terminato) non lascia quindi un file corrotto che verrebbe
  scambiato per uno valido;
- tile non disponibili per una certa zona (ad es. area di mare aperto,
  fuori dalla copertura del DEM) vengono semplicemente saltate, sia
  durante una simulazione normale sia durante un pre-download.

È possibile anche **pre-scaricare** in blocco tutte le tile che coprono
un'area estesa (ad esempio l'intero territorio italiano) prima di
eseguire qualunque simulazione, così che le simulazioni successive su
quell'area partano subito, senza attese di rete durante la fase di
preparazione dati. Una volta in cache, le tile restano disponibili
finché non vengono cancellate manualmente dalla cartella di cache: non
scadono automaticamente.

## 9. Punti di interesse (POI) da OpenStreetMap

Oltre alla mappa dell'incendio, PROPAGATOR può mostrare quali
infrastrutture e strutture critiche dell'area si trovano sul percorso del
fuoco, e quando l'incendio le raggiunge (nella simulazione), riportando
questa informazione come "beni a rischio".

### Da dove vengono scaricati e come

I POI provengono da **OpenStreetMap**, tramite l'**Overpass API**
pubblica (un servizio di interrogazione del database di OpenStreetMap).
L'app invia un'unica interrogazione (query Overpass QL) che copre lo
stesso rettangolo geografico usato per DEM e copertura del suolo, e
raccoglie tutti gli elementi che rientrano nelle categorie note
(indipendentemente dai filtri di categoria scelti dall'utente: il
filtraggio avviene dopo, lato app &mdash; vedi sotto). Poiché l'endpoint
ufficiale (`overpass-api.de`) può risultare non raggiungibile o soggetto
a limiti di traffico da alcune reti, l'app usa di default un mirror
alternativo dello stesso servizio ufficiale; l'endpoint può essere
sostituito, se necessario, tramite una variabile d'ambiente del server
(`PROPAGATOR_OVERPASS_URL`), ad esempio per puntare a un'istanza
self-hosted.

La risposta di Overpass viene anch'essa **salvata in cache su disco**
(sotto `osm/` nella stessa cartella di cache usata per DEM e
WorldCover), identificata da un hash del testo della query: per la
stessa area, richieste successive non generano nuovo traffico verso
Overpass. In caso di errore di rete o di risposta non riuscita, la
richiesta viene ripetuta automaticamente alcune volte con un'attesa
crescente tra un tentativo e l'altro, prima di segnalare un errore.

### Categorie di POI disponibili

Ogni elemento OpenStreetMap restituito viene classificato in una delle
seguenti categorie, in base ai suoi tag:

| Categoria             | Cosa include (tag OpenStreetMap)                                             |
| ---------------------- | ------------------------------------------------------------------------------ |
| Ospedali               | `amenity=hospital`                                                             |
| Caserme dei vigili del fuoco | `amenity=fire_station`                                                   |
| Polizia                | `amenity=police`                                                               |
| Scuole                 | `amenity=school`                                                               |
| Altre emergenze        | Qualunque elemento con un tag `emergency` non già coperto dalle categorie sopra |
| Strade principali      | `highway` di tipo autostrada, strada di grande comunicazione, primaria o secondaria |
| Edifici                | Qualunque elemento con un tag `building`                                       |
| Infrastrutture elettriche | Qualunque elemento con un tag `power` (linee, sottostazioni, centrali, tralicci...); tutti i sottotipi sono raggruppati in questa unica categoria mostrata all'utente |

Per le infrastrutture elettriche e per le strade/edifici estesi (way o
relation), viene conservata &mdash; quando disponibile &mdash; la geometria
completa (l'intero percorso della linea o il contorno del poligono), non
solo un punto rappresentativo: questo permette di verificare quando il
fronte del fuoco raggiunge un qualunque punto lungo l'intera
infrastruttura, non solo il suo centro.

### Come l'utente seleziona le categorie

Nel pannello **Punti di interesse** della web app, l'utente può:

- attivare o disattivare del tutto la ricerca dei POI (casella **Punti di
  interesse**);
- impostare un tetto massimo al numero di POI scaricati/mostrati per
  l'area (**Max POI**, di default 1000): se gli elementi trovati sono di
  più, vengono mantenuti prima quelli delle categorie a priorità più alta
  (ospedali e caserme dei vigili del fuoco per primi, poi polizia/scuole/
  altre emergenze, poi strade, infine edifici) e, a parità di categoria,
  quelli più vicini al centro dell'area;
- selezionare, tramite caselle di spunta indipendenti, quali categorie
  includere tra ospedali, caserme dei vigili del fuoco, polizia, scuole,
  altre emergenze, strade principali, edifici e infrastrutture
  elettriche (tutte selezionate di default).

Il filtro di categoria scelto dall'utente si applica **dopo**
l'interrogazione a Overpass e dopo la lettura della cache: la query stessa scarica
sempre tutte le categorie note (così la cache su disco resta valida anche
se in seguito l'utente cambia selezione), e solo in un secondo momento
l'app tiene o scarta i risultati in base alle categorie scelte, prima di
applicare il tetto massimo **Max POI**.

### Come vengono usati nella simulazione

Per ogni POI mostrato, l'app confronta la sua posizione (o, per le
infrastrutture con geometria estesa, ogni punto lungo il suo percorso)
con la mappa di probabilità di incendio calcolata a ogni fotogramma
temporale, e riporta se e quando il fronte del fuoco lo ha raggiunto.
Sulla mappa, un pallino colorato accanto a ciascun POI distingue quelli
non ancora raggiunti da quelli già raggiunti dall'incendio, aggiornandosi
mentre si scorre lo slider **Tempo**.

## 10. Usare l'API direttamente

Tutto ciò che fa l'interfaccia web passa attraverso una comune API REST,
che puoi chiamare direttamente da uno script, da `curl` o da qualsiasi
client HTTP — utile per automatizzare le esecuzioni o integrare PROPAGATOR
in un altro strumento. Il server **non ha login** ed esegue **un solo
lavoro di simulazione alla volta** per l'intero server (una seconda
richiesta mentre una è in esecuzione riceve un HTTP 429), quindi trattalo
come uno strumento condiviso, su rete fidata, non come un servizio
multi-tenant.

Un esploratore OpenAPI interattivo è sempre disponibile su `/docs`
(Swagger UI) e `/redoc`, con lo schema completo di richiesta/risposta su
`/openapi.json`.

### Endpoint

| Metodo | Percorso                                       | Scopo                                                                                    |
| ------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| POST   | `/api/simulate`                                 | Avvia un nuovo lavoro di simulazione; restituisce `{"id": "..."}` (HTTP 202)              |
| GET    | `/api/simulate/{job_id}`                        | Stato/avanzamento del lavoro (pending, preparing_data, running, done, failed, cancelled)   |
| GET    | `/api/simulate/{job_id}/frames`                 | Confini e lista dei fotogrammi temporali disponibili, con la loro cronologia statistica    |
| GET    | `/api/simulate/{job_id}/frame/{time_s}`         | Isocrone (coordinate in stile GeoJSON) e statistiche per un fotogramma                     |
| GET    | `/api/simulate/{job_id}/frame/{time_s}/image.png` | La mappa di calore della probabilità di incendio per quel fotogramma, come PNG           |
| POST   | `/api/simulate/{job_id}/cancel`                 | Richiede l'annullamento di un lavoro in esecuzione                                         |
| DELETE | `/api/simulate/{job_id}`                        | Rimuove dallo stato del server un lavoro terminato                                         |

### Corpo della richiesta

`POST /api/simulate` accetta un corpo JSON con gli stessi campi descritti
nella sezione 4 (`center_lat`, `center_lon`, `radius_km`, `cellsize`,
`ignition_lat`, `ignition_lon`, `wind_dir`, `wind_speed`, `moisture`,
`realizations`, `do_spotting`, `time_limit_h`, `time_resolution_h`,
`isochrone_thresholds`, e un elenco opzionale `actions` — vedi la sezione 7
per i campi dell'azione). Le stesse limitazioni descritte in questo manuale
sono applicate lato server: una combinazione fuori intervallo o
eccessivamente onerosa viene rifiutata con HTTP 422 prima ancora che venga
creato un lavoro.

### Esempio: eseguire una simulazione end-to-end con curl

```bash
BASE=http://127.0.0.1:8765

# 1. Avvia il lavoro
JOB_ID=$(curl -s -X POST "$BASE/api/simulate" \
  -H "Content-Type: application/json" \
  -d '{
        "center_lat": 44.40, "center_lon": 8.93,
        "radius_km": 10, "cellsize": 30,
        "ignition_lat": 44.42, "ignition_lon": 8.95,
        "wind_dir": 45, "wind_speed": 25, "moisture": 10,
        "realizations": 10, "time_limit_h": 6,
        "time_resolution_h": 1
      }' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 2. Interroga lo stato finché non è completato
until [ "$(curl -s $BASE/api/simulate/$JOB_ID | python3 -c \
    "import sys,json; print(json.load(sys.stdin)['status'])")" = "done" ]; do
  sleep 5
done

# 3. Elenca i fotogrammi disponibili
curl -s "$BASE/api/simulate/$JOB_ID/frames"

# 4. Scarica la mappa di calore dell'ultimo fotogramma
curl -s "$BASE/api/simulate/$JOB_ID/frame/21600/image.png" -o frame.png
```

## 11. Limiti del modello

PROPAGATOR si basa su dati reali (rilievo del terreno, copertura del
suolo) e su modelli del comportamento del fuoco calibrati su osservazioni
ed esperienza, ma resta una semplificazione della realtà: non conosce
condizioni molto locali (una chiazza di combustibile particolarmente
secca o densa, una raffica di vento improvvisa e non prevista), e anche i
dati di input stessi (mappa della vegetazione, rilievo del terreno)
portano un certo margine di approssimazione.

Per questo motivo, i risultati vanno letti come **probabilità e
tendenze**, utili per farsi un'idea di dove e con quale velocità un
incendio potrebbe evolvere in un dato scenario meteorologico, e come
supporto per chi deve decidere dove concentrare attenzione o risorse — non
come una previsione esatta, né come unica base per decisioni operative.
