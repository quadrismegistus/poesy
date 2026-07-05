# Poesy

## Poetic processing, for Python ##

Code developed in the Stanford Literary Lab's "Transhistorical Poetry Project" by Ryan Heuser (@quadrismegistus), J.D. Porter, Jonathan Sensenbaugh, Justin Tackett, Mark Algee-Hewitt, and Maria Kraxenberger. Cleaned and modified from [its original form](http://github.com/quadrismegistus/litlab-poetry) in 2018.

Poesy is built on [Prosodic](http://github.com/quadrismegistus/prosodic), a metrical-phonological parser written in Python.

> **What's new in 0.4:** poesy is now a thin compatibility layer over **prosodic v3**, which absorbed poesy's poem-level analysis (meter-type classification, line/syllable scheme detection, rhyme-scheme matching, sonnet detection) into its [`prosodic.analysis`](https://github.com/quadrismegistus/prosodic) module. The classic `Poem` API below is preserved; the heavy lifting happens in prosodic. If you're starting fresh, consider using prosodic v3 directly (`text.meter_type`, `text.rhyme_scheme`, `text.is_sonnet`, `text.summary()`); if you have poesy code, it should keep working. The underlying `prosodic.TextModel` is always available as `poem.text`.

## Installation

### 1. Install espeak

Install espeak, free TTS software, to 'sound out' unknown words. See [here](http://espeak.sourceforge.net/download.html) for all downloads.

* On Linux: ```apt-get install espeak```

* On Mac:
  * Install [homebrew](https://brew.sh) if not already installed.

  * Type into the Terminal app: `brew install espeak`

* On Windows:
        Download and install from http://espeak.sourceforge.net/download.html.

### 2. Install Poesy

```
pip install poesy
```

Or the development version:

```
pip install -U git+https://github.com/quadrismegistus/poesy
```

## Usage

### Create a poem: `poem = Poem()`

```python
from poesy import Poem

# create a Poem object by string
poem = Poem("""
When in the chronicle of wasted time
I see descriptions of the fairest wights,
And beauty making beautiful old rhyme
In praise of ladies dead and lovely knights,
Then, in the blazon of sweet beauty's best,
Of hand, of foot, of lip, of eye, of brow,
I see their antique pen would have express'd
Even such a beauty as you master now.
So all their praises are but prophecies
Of this our time, all you prefiguring;
And, for they look'd but with divining eyes,
They had not skill enough your worth to sing:
For we, which now behold these present days,
Had eyes to wonder, but lack tongues to praise.
""")

# or create a Poem object by pointing to a text file
la_belle_dame = Poem(fn='poems/keats.la_belle_dame_sans_merci.txt')
```

### Summary of annotations: `poem.summary()`

A quick tabular summary of most of the annotations Poesy has made on the poem.

      (#s,#l)  parse                                             rhyme      #feet    #syll    #parse
    ---------  ------------------------------------------------  -------  -------  -------  --------
         1.1   when|IN|the|CHRO|ni|CLE*|of|WA|sted|TIME          a              5       10         1
         1.2   i|SEE|de|SCRIP|tions|OF*|the|FAI|rest|WIGHTS      b              5       10         1
         1.3   and|BEAU|ty|MA|king|BEAU|ti|FUL*|old*|RHYME       a              5       10         3
         1.4   in|PRAISE|of|LA|dies|DEAD|and|LOVE|ly|KNIGHTS     b              5       10         1
         1.5   then|IN|the|BLA|zon|OF*|sweet*|BEAU|ty's|BEST     -              5       10         4
         1.6   of|HAND|of|FOOT|of|LIP|of|EYE|of|BROW             c              5       10         1
         1.7   i|SEE|their.an*|TIQUE.PEN*|would|HAVE|e|XPRESS'D  -              4       10         6
         1.8   E|ven|SUCH*|a|BEAU|ty|AS*|you|MA|ster|NOW         c              6       11         2
         1.9   so|ALL|their|PRAI|ses|ARE*|but|PRO|phe.cies*      -              4       10         3
         1.1   of|THIS*|our|TIME|all|YOU*|pre|FI|gu.ring*        d              4       10         6
         1.11  and|FOR*|they|LOOK'D|but|WITH*|di|VI|ning|EYES    e              5       10         2
         1.12  they|HAD|not|SKILL|e|NOUGH|your|WORTH|to|SING     d              5       10         1
         1.13  for|WE*|which|NOW|be|HOLD|these|PRE|sent|DAYS     e              5       10         2
         1.14  had|EYES|to|WON|der|BUT*|lack*|TONGUES|to|PRAISE  e              5       10         3


    estimated schema
    ----------
    meter: Iambic
    feet: Pentameter
    syllables: 10
    rhyme: Sonnet A (abab cdcd eefeff)

### Statistics on annotations: `poem.statd`

This dictionary combines the following dictionaries.

#### 1. Estimated line scheme (in feet): `poem.schemed_beat`

```
{'scheme': (5,),
 'scheme_diff': 8,
 'scheme_length': 1,
 'scheme_repr': 'Pentameter',
 'scheme_type': 'Invariable'}
```

#### 2. Estimated line scheme (in syllables): `poem.schemed_syll`

```
{'scheme': (10,),
 'scheme_diff': 1,
 'scheme_length': 1,
 'scheme_repr': 10,
 'scheme_type': 'Invariable'}
```

#### 3. Estimated metrical scheme: `poem.meterd`

```
{'ambiguity': 2.0937047867021317,
 'constraint_TOTAL': 0.14598540145985403,
 'constraint_s_unstress': 0.0948905109489051,
 'constraint_unres_across': 0.014598540145985401,
 'constraint_unres_within': 0.014598540145985401,
 'constraint_w_stress': 0.021897810218978103,
 'length_avg_line': 10.071428571428571,
 'length_avg_parse': 10.071428571428571,
 'mpos_s': 0.48905109489051096,
 'mpos_ss': 0.0072992700729927005,
 'mpos_w': 0.48175182481751827,
 'mpos_ww': 0.021897810218978103,
 'perc_lines_ending_s': 0.8571428571428571,
 'perc_lines_ending_w': 0.14285714285714285,
 'perc_lines_fourthpos_s': 0.8571428571428571,
 'perc_lines_fourthpos_w': 0.14285714285714285,
 'perc_lines_starting_s': 0.07142857142857142,
 'perc_lines_starting_w': 0.9285714285714286,
 'type_foot': 'binary',
 'type_head': 'final',
 'type_scheme': 'iambic'}
```

Note: as of 0.4, the `constraint_*` keys use prosodic v3's constraint names, which differ from prosodic v1's (e.g. `s_unstress` rather than `stress_s=>-u`).

#### 4. Estimated rhyme scheme: `poem.rhymed`

```
{'rhyme_scheme': ('Sonnet A', 'abab cdcd eefeff'),
 'rhyme_scheme_accuracy': 0.7,
 'rhyme_scheme_form': 'abab cdcd eefeff',
 'rhyme_scheme_name': 'Sonnet A',
 'rhyme_schemes': [(('Sonnet A', 'abab cdcd eefeff'), 0.7),
                   (('Sonnet, Shakespearean', 'abab cdcd efefgg'),
                    0.5555555555555556),
                   (('Sonnet E', 'abab cbcd cdedee'), 0.42857142857142855),
                   (('Sonnet B', 'abab cdcd effegg'), 0.4),
                   (('Sonnet D', 'ababbcdc ceceff'), 0.35714285714285715)]}
```

### Iterate over lines: `poem.lined`

Every poem has a number of dictionaries, each keyed to a "line ID", a tuple of `(linenum, stanzanum)`.

```python
# The dictionary storing the string representation for the line:
for lineid,line_str in sorted(poem.lined.items()):
    print(lineid,line_str)

# Use this dictionary to loop over prosodic's Line objects instead
# (as of 0.4 these are prosodic v3 Line objects)
for lineid,line_obj in sorted(poem.prosodic.items()):
    print(lineid,line_obj.best_parse)

# Other dictionaries
poem.linenums              # lineid -> line number within poem
poem.linenums_bystanza     # lineid -> line number within stanza
poem.stanzanums            # lineid -> stanza number
poem.linelengths           # lineid -> length of line (in syllables)
poem.linelengths_bybeat    # lineid -> length of line (in feet)
poem.numparses             # lineid -> number of plausible parses for line
poem.rhymes                # lineid -> rhyme scheme symbol
```

## Configure

Poesy depends on [Prosodic](http://github.com/quadrismegistus/prosodic) (v3) for metrical parsing.

Prosodic v3 no longer uses named meter-config files (a v1 feature); meters are configured by keyword arguments instead (see `prosodic.Meter`). Pass a dict of meter kwargs to a Poem object or its parse method:

```python
from poesy import Poem
poem = Poem(fn='poems/shakespeare_sonnets/sonnet-001.txt',
            meter=dict(max_s=2, max_w=2))
```

Or:

```python
poem.parse(max_s=2, max_w=2)
```

Passing a v1 meter name string (e.g. `meter='iambic_pentameter'`) emits a warning and falls back to prosodic's default meter.

## Migrating to prosodic v3

Everything poesy computes is available directly in prosodic v3:

| poesy | prosodic v3 |
| --- | --- |
| `poem.summary()` | `text.summary()` |
| `poem.statd['meter_type_scheme']` | `text.meter_type['type']` |
| `poem.schemed_beat` | `text.line_scheme` |
| `poem.schemed_syll` | `text.syllable_scheme` |
| `poem.rhymed` | `text.rhyme_scheme` |
| `poem.rime_ids` | `text.rhyme_ids` |
| `poem.isSonnet` | `text.is_sonnet` |
| `poem.isShakespeareanSonnet` | `text.is_shakespearean_sonnet` |
