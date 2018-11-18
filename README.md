# Poetix

## Poetic processing, for Python ##

```python
from poetix import Poem

sonnet = Poem(u"""
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
Had eyes to wonder, but lack tongues to praise.""")

# See how poetix understood the poem
sonnet.summary()
```

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

```python
# Get all the statistics
sonnet.statd
```

  {'beat_scheme': (5,),
   'beat_scheme_diff': 6,
   'beat_scheme_length': 1,
   'beat_scheme_repr': 'Pentameter',
   'beat_scheme_type': 'Invariable',
   'meter_ambiguity': 3.5,
   'meter_constraint_TOTAL': 0.14285714285714285,
   'meter_constraint_footmin-f-resolution': 0.022556390977443608,
   'meter_constraint_footmin-w-resolution': 0.0,
   'meter_constraint_strength_w=>-p': 0.0,
   'meter_constraint_stress_s=>-u': 0.09774436090225563,
   'meter_constraint_stress_w=>-p': 0.022556390977443608,
   'meter_length_avg_line': 10.071428571428571,
   'meter_length_avg_parse': 10.071428571428571,
   'meter_mpos_s': 0.5037593984962406,
   'meter_mpos_ss': 0.015037593984962405,
   'meter_mpos_w': 0.43609022556390975,
   'meter_mpos_ww': 0.045112781954887216,
   'meter_perc_lines_ending_s': 1.0,
   'meter_perc_lines_fourthpos_s': 0.8571428571428571,
   'meter_perc_lines_fourthpos_w': 0.14285714285714285,
   'meter_perc_lines_starting_s': 0.35714285714285715,
   'meter_perc_lines_starting_w': 0.6428571428571429,
   'meter_type_foot': 'binary',
   'meter_type_head': 'final',
   'meter_type_scheme': 'iambic',
   'num_lines': 14,
   'rhyme_scheme': ('Sonnet, Shakespearean', 'abab cdcd efefgg'),
   'rhyme_scheme_accuracy': 0.6363636363636364,
   'rhyme_scheme_form': 'abab cdcd efefgg',
   'rhyme_scheme_name': 'Sonnet, Shakespearean',
   'rhyme_schemes': [(('Quatrain And Triplet', 'ababccc'), 0.4),
    (('Sonnet C', 'ababacdc edefef'), 0.4),
    (('Sonnet E', 'abab cbcd cdedee'), 0.4117647058823529),
    (('Sonnet A', 'abab cdcd eefeff'), 0.6153846153846154),
    (('Sonnet, Shakespearean', 'abab cdcd efefgg'), 0.6363636363636364)],
   'syll_scheme': (10,),
   'syll_scheme_diff': 1,
   'syll_scheme_length': 1,
   'syll_scheme_repr': 10,
   'syll_scheme_type': 'Invariable'}


## Research context

Code used in the [Literary Lab](http://litlab.stanford.edu)'s [Trans-historical Poetry Project](http://litlab.stanford.edu/?page_id=13), involving myself ([Ryan Heuser](http://twitter.com/quadrismegistus)), [Mark Algee-Hewitt](https://twitter.com/mark_a_h), Maria Kraxenberger, J.D. Porter, Jonny Sensenbaugh, and Justin Tackett. We presented the project at DH2014 in Lausanne. The abstract is [here](http://dharchive.org/paper/DH2014/Paper-788.xml), but a better source of information is our slideshow (with notes) [here](https://docs.google.com/presentation/d/1KyCi4s6P1fE4D3SlzlZPnXgPjwZvyv_Vt-aU3tlb24I/edit?usp=sharing). The project has been going on for 2+ years, and we are currently in the process of drafting up the project as an article for submission to a journal. Feel free to use this code for any purpose whatever, but please provide attribution back to (for now) this webpage and the aforementioned authors.

The goal in the project is to develop software capable of annotating the following four features of poetic form:

1. Stanzaic scheme (Syllable scheme / beat scheme) [**Complete**]:
  * An example scheme is: _10_ (Invariable) or _8-6_ (Alternating) or _10-10-10-10-10-6_ (Complex)
  * Invariable schemes (e.g. Inv_10 = the poem is generally always in lines of 10 syllables in length, e.g. blank verse, sonnets, heroic couplets)
  * Alternating schemes (e.g. _Alt_8_6_ = the poem alternates between lines of 8 and 6 syllables in length. Most common in ballads)
  * Complex schemes (basically, everything more complex than the above two. Includes odes, free verse, etc)

2. Metrical scheme [**Complete**]:
  * Produce a scansion of each of the poem's lines, and then decide if the poem's meter is predominantly:
    1. Iambic (Binary foot, head final)
    2. Trochaic (Binary foot, head initial)
    3. Anapestic (Ternary foot, head final)
    4. Dactylic (Ternary foot, head initial)


Code developed in the Stanford Literary Lab by Ryan Heuser(@quadrismegistus), J.D. Porter, Jonathan Sensenbaugh, Justin Tackett, Mark Algee-Hewitt, and Maria Kraxenberger.

Cleaned and modified in 2018. Original code is available [here](http://github.com/quadrismegistus/litlab-poetry).
