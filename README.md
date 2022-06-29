# Poesy

## Poetic processing, for Python ##

Code developed in the Stanford Literary Lab's "Transhistorical Poetry Project" by Ryan Heuser (@quadrismegistus), J.D. Porter, Jonathan Sensenbaugh, Justin Tackett, Mark Algee-Hewitt, and Maria Kraxenberger. Cleaned and modified from [its original form](http://github.com/quadrismegistus/litlab-poetry) in 2018.

Poesy is built on [Prosodic](http://github.com/quadrismegistus/prosodic), a metrical-phonological parser written in Python.

## Demo

For a demo of poesy's installation and in action, see [this Colab notebook](https://colab.research.google.com/drive/1pl0qY8bi-QD_peC2mQVNCG4lxICxImwy?usp=sharing).

## Installation

### 1. Insteall espeak

Install espeak, free TTS software, to 'sound out' unknown words. See [here](http://espeak.sourceforge.net/download.html) for all downloads.

* On Linux: ```apt-get install espeak```
    
* On Mac:
  * Install [homebrew](brew.sh) if not already installed.

  * Type into the Terminal app: `brew install espeak`
    
* On Windows:
        Download and install from http://espeak.sourceforge.net/download.html.

### 2. Install Poesy

Install:

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
         1.1   WHEN|in.the|CHRON|i|CLE*|of|WAST|ed|TIME          a              5       10         2
         1.2   i|SEE|de|SCRIP|tions|OF*|the|FAI|rest|WIGHTS      b              5       10         1
         1.3   and|BEAU|ty|MAK|ing|BEAU|ti|FUL*|old*|RHYME       a              5       10         3
         1.4   in|PRAISE|of|LAD|ies|DEAD|and|LOVE|ly|KNIGHTS     b              5       10         1
         1.5   THEN|in.the|BLA|zon|OF*|sweet*|BEAU|tys|BEST      c              5       10         8
         1.6   of|HAND|of|FOOT|of|LIP|of|EYE|of|BROW             d              5       10         1
         1.7   i|SEE|their.an*|TIQUE.PEN*|would|HAVE|ex|PRESSD   c              4       10         5
         1.8   EV|en|SUCH*|a|BEAU|ty|AS*|you|MAS|ter|NOW         d              6       11         1
         1.9   so|ALL|their|PRAIS|es|ARE*|but|PRO|phe|CIES*      e              5       10         3
         1.1   OF*|this.our|TIME|all|YOU*|pre|FIG|ur|ING*        f              5       10        15
         1.11  and.for|THEY.LOOKD*|but|WITH*|di|VIN|ing|EYES     e              4       10         3
         1.12  THEY|had.not|SKILL|en|OUGH|your|WORTH|to|SING     f              5       10         2
         1.13  for|WE|which|NOW|be|HOLD|these|PRE|sent|DAYS      e              5       10         1
         1.14  had|EYES|to|WON|der|BUT*|lack*|TONGUES|to|PRAISE  e              5       10         3


    estimated schema
    ----------
    meter: Iambic
    feet: Pentameter
    syllables: 10
    rhyme: Sonnet, Shakespearean (abab cdcd efefgg)

### Statistics on annotations: `poem.statd`

This dictionary combines the following dictionaries.

#### 1. Estimated line scheme (in feet): `poem.schemed_beat`

```
{'scheme': (5,),
 'scheme_diff': 2,
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
{'ambiguity': 3.5,
 'constraint_TOTAL': 0.14285714285714285,
 'constraint_footmin-f-resolution': 0.007142857142857143,
 'constraint_footmin-w-resolution': 0.0,
 'constraint_strength_w=>-p': 0.0,
 'constraint_stress_s=>-u': 0.10714285714285714,
 'constraint_stress_w=>-p': 0.02857142857142857,
 'length_avg_line': 10.071428571428571,
 'length_avg_parse': 10.071428571428571,
 'mpos_s': 0.5,
 'mpos_ss': 0.007142857142857143,
 'mpos_w': 0.4928571428571429,
 'perc_lines_ending_s': 1.0,
 'perc_lines_fourthpos_s': 0.8571428571428571,
 'perc_lines_fourthpos_w': 0.14285714285714285,
 'perc_lines_starting_s': 0.14285714285714285,
 'perc_lines_starting_w': 0.8571428571428571,
 'type_foot': 'binary',
 'type_head': 'final',
 'type_scheme': 'iambic'}
```

#### 4. Estimated rhyme scheme: `poem.rhymed`

```
{'rhyme_scheme': ('Sonnet, Shakespearean', 'abab cdcd efefgg'),
 'rhyme_scheme_accuracy': 0.6363636363636364,
 'rhyme_scheme_form': 'abab cdcd efefgg',
 'rhyme_scheme_name': 'Sonnet, Shakespearean',
 'rhyme_schemes': [(('Sonnet, Shakespearean', 'abab cdcd efefgg'),
   0.6363636363636364),
  (('Sonnet A', 'abab cdcd eefeff'), 0.6153846153846154),
  (('Sonnet E', 'abab cbcd cdedee'), 0.4117647058823529),
  (('Quatrain And Triplet', 'ababccc'), 0.4),
  (('Sonnet C', 'ababacdc edefef'), 0.4)]}
```

### Iterate over lines: `poem.lined`

Every poem has a number of dictionaries, each keyed to a "line ID", a tuple of `(linenum, stanzanum)`.

```python
# The dictionary storing the string representation for the line:
for lineid,line_str in sorted(poem.lined.items()):
    print(lineid,line_str)
    
# Use this dictionary to loop over prosodic's Line objects instead
for lineid,line_obj in sorted(poem.prosodic.items()):
    print(lineid,line_obj.bestParse())
    
# Other dictionaries
poem.linenums              # lineid -> line number within poem
poem.linenums_bystanza     # lineid -> line number within stanza
poem.stanzanums            # lineid -> stanza number
poem.linelengths           # lineid -> length of line
poem.linelengths_bybeat    # lineid -> length of line (in feet)
poem.numparses             # lineid -> number of plausible parses for line
poem.rhymes                # lineid -> rhyme scheme symbol
```


## Configure

Poesy depends on [Prosodic](http://github.com/quadrismegistus/prosodic) for metrical parsing. Prosodic stores its configuration data in `~/prosodic_data/`; the `README.txt` there has more information.

By default, Poesy will use `~/prosodic_data/meters/meter_default.py` as its meter (its set of metrical constraints and behaviors). Open that file to read more details.

To specify a different meter, pass a meter name to a Poem object:

```python
from poesy import Poem
poem = Poem(fn='poems/shakespeare_sonnets/sonnet-001.txt',
            meter='iambic_pentameter')
```
Or to the parse method:

```python
poem.parse(meter='iambic_pentameter')
```

These will load the meter in `~/prosodic_data/meters/iambic_pentameter.py`.
