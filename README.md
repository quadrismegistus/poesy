# Poetix

## Poetic processing, for Python ##





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
