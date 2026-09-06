# Data: every external input, where it comes from, and where to put it

The engine and the tests need no data. The mechanism experiments e7, e8, e9, e12 and e13
generate their own streams, and so do two of the three streams e5 runs, whose third is the
Wikipedia cache described in section 4 and committed here. Five inputs come from outside the
repository, and this file is the source for each one.

Every path below is relative to `ESCROW_ROOT`. Scripts read that environment variable and fall
back to the repository root, which is the directory holding `code/`, `results/` and `paper/`:

```bash
export ESCROW_ROOT=/path/to/escrow     # optional; the repository root by default
```

Each URL below was checked with `curl -sI -L` on 2026-09-05. Where a size is given, it is the
`Content-Length` the server returned or the size the GitHub tree API reported. If a source
could not be checked, this file says "source not verified" and gives no URL.

Software dependencies are in `requirements.txt`. None of the files below are installed by pip.

---

## 1. MusicBrainz 20K

**What it is.** One CSV of 19,375 records about songs, pooled from five sources, with 10,000
true clusters and 16,250 duplicate pairs. The columns are TID, CID, CTID, SourceID, id, number,
title, length, artist, album, year and language. CID is the cluster id, which is the ground
truth. The duplicates were made by the DAPO data generator, not by nature, so the records are
real MusicBrainz songs with generated corruptions. `mb20k_benchmark.py` reads it and never lets
CID reach any method's input.

**Where it comes from.** The database group of Prof. Erhard Rahm at Leipzig University publishes
it on the page "Benchmark datasets for entity resolution", in the multi-source (FAMER) family.
The exact name of the file is `musicbrainz-20-A01.csv.dapo`, listed on that page as
"Music Brainz 20K".

Verified:

- landing page: `https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution`
  returns HTTP 200. The older address
  `https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution`
  returns HTTP 301 to it.
- the file: `https://dbs.uni-leipzig.de/files/datasets/saeedi/musicbrainz-20-A01.csv.dapo`
  returns HTTP 200, `Content-Type: text/csv`, `Content-Length: 2170333`.
- the column readme: `https://dbs.uni-leipzig.de/files/datasets/saeedi/musicBrainz_readme.txt`
  returns HTTP 200, 743 bytes. It is where the meaning of TID, CID, CTID and SourceID above
  comes from.

**Expected path.**

```
$ESCROW_ROOT/data/musicbrainz/musicbrainz-20-A01.csv.dapo
```

**Size.** About 2.1 MB (2,170,333 bytes).

**Licence and terms.** The page states, under the multi-source table: "These datasets are made
available by the database group of Prof. Erhard Rahm under the Creative Commons license", and
links `https://creativecommons.org/licenses/by/4.0/` (CC BY 4.0, verified HTTP 200). The page
asks that credit refer to the website and, for the multi-source datasets, to the ADBIS 2017
paper it cites.

**How to get it.**

```bash
mkdir -p "$ESCROW_ROOT/data/musicbrainz"
curl -L -o "$ESCROW_ROOT/data/musicbrainz/musicbrainz-20-A01.csv.dapo" \
  https://dbs.uni-leipzig.de/files/datasets/saeedi/musicbrainz-20-A01.csv.dapo
```

---

## 2. CESI: the ReVerb45K test split and `metrics.py`

**What it is.** Two files from the CESI repository, the code release of the WWW 2018 paper on
canonicalising open knowledge bases.

- `reverb45k_test`: the test split of the ReVerb45K benchmark, one JSON object per line, with
  the raw triple, the authors' own normalised triple (`triple_norm`), the gold entity link
  (`true_link`) and a mention id. `e10_reverb45k.py` builds one record per unique normalised
  subject string from it, 12,327 records over 36,814 mentions and 6,061 gold clusters.
- `src/metrics.py`: CESI's own scoring code. We score with it, not with a re-implementation, so
  that macro, micro and pairwise F1 mean what the CESI paper means. The script puts
  `baselines/cesi/src` on `sys.path` and does `from metrics import evaluate`.

**Where it comes from.** `github.com/malllabiisc/cesi`, branch `master`.

Verified:

- repository: `https://github.com/malllabiisc/cesi` returns HTTP 200. The GitHub API reports
  default branch `master` and licence `Apache-2.0`.
- test split: `https://raw.githubusercontent.com/malllabiisc/cesi/master/data/reverb45k/reverb45k_test`
  returns HTTP 200, `Content-Length: 36734628`.
- metrics: `https://raw.githubusercontent.com/malllabiisc/cesi/master/src/metrics.py`
  returns HTTP 200, `Content-Length: 4095`.

The repository also holds `data/reverb45k/reverb45k_valid` (8,373,083 bytes). No script here
reads it.

**Expected paths.**

```
$ESCROW_ROOT/baselines/cesi/data/reverb45k/reverb45k_test
$ESCROW_ROOT/baselines/cesi/src/metrics.py
```

The simplest way to get both is to clone the repository into `baselines/cesi`.

**Size.** The test split is about 35 MB (36,734,628 bytes). `metrics.py` is 4 KB. A full clone
is larger, because the repository also ships `data/base.zip`, `data/ambiguous.zip` and the PPDB
files.

**Licence and terms.** Apache License 2.0, stated by the repository's `LICENSE` file and by the
GitHub API.

**One deviation to know about.** CESI's own entity normalisation (`proc_ent`) does not run on
Python 3.12, so every method in `e10_reverb45k.py` uses the dataset's own `triple_norm` field
instead. This is recorded in the `deviations` field of the results file.

**How to get it.**

```bash
mkdir -p "$ESCROW_ROOT/baselines"
git clone https://github.com/malllabiisc/cesi "$ESCROW_ROOT/baselines/cesi"
```

---

## 3. AutoPKG Lazada CSVs

**What it is.** Three CSVs from a real marketplace catalogue, released with the AutoPKG paper
(Hongwimol et al., "AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute
Knowledge Graph Construction", arXiv:2604.16950, Findings of ACL 2026). `lazada_c2_rawkey.py`
and the Lazada arm of `llm_graph_formation.py` read them.

- `lazada_autopkg_product_data_url.csv`: about 36,000 Lazada (Philippines) products with
  `product_id`, `product_name`, `highlight`, `description`, `specifications` and image URLs. We
  read only `specifications`, the published spec map, and keep its keys exactly as published.
  21,365 listings carry a parseable spec map.
- `lazada_autopkg_kg_nodes.csv`: the nodes of AutoPKG's own product knowledge graph, of type
  `Product`, `Type` or `Key`.
- `lazada_autopkg_kg_edges.csv`: the edges of that graph. We read only the `of_type` edges, to
  get a silver product-type label per listing. Those labels are model-derived, they are labelled
  silver everywhere in the paper, and they never reach the engine.

**Where it comes from.** The code and data repository the AutoPKG paper points to.

Verified:

- the paper: `https://arxiv.org/abs/2604.16950` returns HTTP 200, title "AutoPKG: An Automated
  Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction". The full
  text at `https://arxiv.org/html/2604.16950v1` is where the repository link below appears.
- the repository: `https://github.com/Product-Understanding-Lazada-Alibaba/AutoPKG` returns
  HTTP 200. The GitHub API reports default branch `main` and licence `MIT`.
- one file, checked directly:
  `https://raw.githubusercontent.com/Product-Understanding-Lazada-Alibaba/AutoPKG/main/data/lazada_autopkg_product_data_url.csv`
  returns HTTP 200, `Content-Length: 44092331`. The other two sizes below are from the GitHub
  tree API for branch `main`.

The repository also ships `data/lazada_autopkg_product_data_file_path.csv` (71,320,748 bytes),
the same product table with image file paths instead of URLs. No script here reads it.

**Expected paths.**

```
$ESCROW_ROOT/baselines/autopkg/data/lazada_autopkg_product_data_url.csv
$ESCROW_ROOT/baselines/autopkg/data/lazada_autopkg_kg_nodes.csv
$ESCROW_ROOT/baselines/autopkg/data/lazada_autopkg_kg_edges.csv
```

**Size.** `product_data_url.csv` about 42 MB (44,092,331 bytes), `kg_nodes.csv` about 42 MB
(43,584,837 bytes), `kg_edges.csv` about 80 MB (84,145,710 bytes). About 164 MB for the three.
A full clone is larger.

**Licence and terms.** The repository carries the MIT licence. Its README describes the four
data files as derived from the Lazada e-commerce platform (Philippines region) and states no
separate licence for the data. Product images stay on Lazada's own CDN and are referenced by
URL; we never fetch them.

**How to get it.**

```bash
mkdir -p "$ESCROW_ROOT/baselines"
git clone https://github.com/Product-Understanding-Lazada-Alibaba/AutoPKG "$ESCROW_ROOT/baselines/autopkg"
```

---

## 4. Wikipedia infobox caches

**What it is.** Three JSON files, one per category, each a list of raw infobox parameter maps.
`wikipedia_demo.py` fetches the pages of three unrelated categories, parses the first infobox of
each page into its top-level `| key = value` parameters with no cleaning and no schema, and
caches the result. The three files are the Wikipedia stream used by `wikipedia_demo.py`,
`e4_baseline_army.py`, `e5_order_dependence.py` and the Wikipedia arm of
`llm_graph_formation.py`.

| File | Category | Records | Size |
| --- | --- | --- | --- |
| `results/wiki_cache.film.json` | `Category:1994 films` | 120 | 51,330 bytes |
| `results/wiki_cache.person.json` | `Category:English male film actors` | 113 | 31,306 bytes |
| `results/wiki_cache.mountain.json` | `Category:Alpine four-thousanders` | 87 | 49,219 bytes |

**Where it comes from.** The public MediaWiki API of the English Wikipedia,
`https://en.wikipedia.org/w/api.php` (verified HTTP 200), queried by
`code/experiments/wikipedia_demo.py` on its first run. There is nothing to download by hand:
**the three caches are already committed under `results/`**, so every Wikipedia number in the
paper replays offline. Delete a cache file and the demo rebuilds it from the API.

`llm_graph_formation.py` reads these caches from `results/`. Until 2026-09-05 it looked in a
`wiki/` directory that does not exist in this repository; that is fixed.

**Expected paths.**

```
$ESCROW_ROOT/results/wiki_cache.film.json
$ESCROW_ROOT/results/wiki_cache.person.json
$ESCROW_ROOT/results/wiki_cache.mountain.json
```

**Size.** About 130 KB for the three.

**Licence and terms.** Wikipedia article text, which is what an infobox is, is published under
CC BY-SA 4.0 (`https://creativecommons.org/licenses/by-sa/4.0/`, verified HTTP 200). The
MediaWiki API asks callers for a descriptive User-Agent and for polite rate limits
(`https://www.mediawiki.org/wiki/API:Etiquette`, verified HTTP 200). `wikipedia_demo.py` sends
its own User-Agent, passes `maxlag=5`, and backs off on HTTP 429.

---

## 5. The development catalogue CSV (`ESCROW_CATALOG_CSV`)

**What it is.** One CSV export of a raw e-commerce catalogue, with a `department` column and a
`specs` column holding a JSON object of raw spec keys as published. `catalogue_devset.py` reads
it, keeps one department at a time, and writes `results/catalogue_<department>.json`.

**Where it comes from.** A private catalogue supplied by the author. It is **not public**,
there is no download, and **source not verified**: no URL is given here because none exists.

**Expected path.** Whatever the environment variable names. There is no default location inside
the repository; the script falls back to `catalog.csv` in the working directory.

```bash
export ESCROW_CATALOG_CSV=/path/to/catalog.csv
python3 code/experiments/catalogue_devset.py Footwear 8000
```

**Size.** Not stated. The committed runs used 993 footwear records and 12,000 apparel records.

**Licence and terms.** None stated. Treat it as private data. Do not redistribute it, and do not
commit it.

**What depends on it.** No benchmark result. MusicBrainz 20K, ReVerb45K, Lazada, the Wikipedia
demo, the language-model comparison and every mechanism experiment run without this file, and a
reader who cannot get it can still reproduce all of them.

One honest exception: the appendix does print a catalogue exhibit (the "Raw records: a catalogue
department" subsection, and the `catalogue` entry in the results-file walkthrough): 993 footwear
records, 43 raw keys, K = 6 from 51 mint events, 303.4 records per second. Those numbers are
read from a results file that this repository does not ship, because it carries the private
catalogue's own attribute keys and value counts. That one run is therefore the only run in the
paper whose numbers an outside reader can neither repeat nor check. Every other number the paper
prints has its file under `results/` here.

---

## Mentioned in the paper, and not shipped here

The Alaska entity-resolution benchmark (`camera`, `monitor` and `notebook`, about 270 KB each)
is held for the planned key-identity run. No script in `code/` reads it yet, so this repository
does not ship it, and it is named here only so that nobody hunts for a loader that does not
exist. It is published by the DI2KG organisers at `github.com/merialdo/research.alaska` under
the MIT licence.
