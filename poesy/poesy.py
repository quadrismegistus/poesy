## encoding=utf-8
"""
Poesy: poetic processing, for Python.

As of 0.4.0, poesy is a thin compatibility layer over prosodic v3
(https://github.com/quadrismegistus/prosodic), which absorbed poesy's
poem-level analysis — meter-type classification, line/syllable scheme
detection, rhyme-scheme matching, sonnet detection — into its
``prosodic.analysis`` module. The classic ``Poem`` API is preserved here;
the heavy lifting happens in prosodic.

The underlying ``prosodic.TextModel`` is available as ``poem.text`` for
anything this shim doesn't cover.
"""

import logging
import math
import os
import statistics
import warnings
from collections import Counter
from functools import cached_property

logger = logging.getLogger(__name__)

# Constants (kept for backwards compatibility with poesy<=0.3)
PATH_RHYME_SCHEMES = os.path.join(os.path.dirname(__file__), 'schemes', 'rhyme_schemes.txt')
METER = 'default_english'
# Legacy prosodic-v1 knobs. No longer consulted: rhyme detection now uses
# prosodic v3's calibrated feature-edit distance (see
# prosodic.analysis.rhyme_scheme.compute_rhyme_ids).
MAX_RHYME_DIST = 5
PROSODIC_CONFIG = {}

# prosodic-v1 meter names that map onto prosodic v3's default meter
_V1_DEFAULT_METER_NAMES = {'default_english', 'meter_default', 'default'}


def test():
    poemtxt = """Since brass, nor stone, nor earth, nor boundless sea,
	But sad mortality o'er-sways their power,
	How with this rage shall beauty hold a plea,
	Whose action is no stronger than a flower?
	O, how shall summer's honey breath hold out
	Against the wreckful siege of battering days,
	When rocks impregnable are not so stout,
	Nor gates of steel so strong, but Time decays?
	O fearful meditation! where, alack,
	Shall Time's best jewel from Time's chest lie hid?
	Or what strong hand can hold his swift foot back?
	Or who his spoil of beauty can forbid?
	O, none, unless this miracle have might,
	That in black ink my love may still shine bright."""

    p = Poem(poemtxt)
    return p.rhymed


class Poem(object):
    def __init__(self, txt=None, id=None, title=None, fn=None, fn_encoding='utf-8', meter=METER, lang='en'):
        if fn and not txt:
            if os.path.exists(fn):
                with open(fn, encoding=fn_encoding) as f:
                    txt = f.read()

        if not txt:
            raise ValueError("Neither a txt string object was passed nor a working filename through fn=")

        txt = txt.strip()
        txt = txt.replace('\r\n', '\n').replace('\r', '\n')
        while '\n\n\n' in txt:
            txt = txt.replace('\n\n\n', '\n\n')

        self.id = hash(txt) if not id else id
        self.meter = meter
        self._meter_kwargs = self._resolve_meter(meter)
        self.title = title if title else txt.split('\n')[0].strip()
        self.txt = txt
        self.lang = lang
        self.genn = True

        import prosodic
        self.text = prosodic.Text(txt, lang=lang)

    @staticmethod
    def _resolve_meter(meter):
        """Translate a poesy<=0.3 ``meter`` argument into prosodic-v3 parse kwargs.

        prosodic v3 has no named meter-config files; meters are configured by
        keyword arguments to ``TextModel.parse()`` (see ``prosodic.Meter``).
        """
        if meter is None or meter in _V1_DEFAULT_METER_NAMES:
            return {}
        if isinstance(meter, dict):
            return dict(meter)
        if isinstance(meter, str):
            warnings.warn(
                f"meter={meter!r}: named meter configs are a prosodic-v1 feature; "
                "prosodic v3 configures meters by keyword arguments instead "
                "(see prosodic.Meter). Using the default meter; pass a dict of "
                "meter kwargs to customize.",
                stacklevel=3,
            )
            return {}
        # assume a prosodic Meter instance (or something parse() understands)
        return {'meter': meter}

    ## Lines and stanzas

    @cached_property
    def lined(self):
        """Dictionary of lineid -> line text, where lineid is (linenum, stanzanum)."""
        d = {}
        for stanza in self.text.stanzas:
            for line in stanza.lines:
                d[(line.num, stanza.num)] = line.txt.strip()
        return d

    @cached_property
    def prosodic(self):
        """Dictionary of lineid -> prosodic (v3) Line object."""
        d = {}
        for stanza in self.text.stanzas:
            for line in stanza.lines:
                d[(line.num, stanza.num)] = line
        return d

    @property
    def lines(self):
        return [v for k, v in sorted(self.lined.items())]

    @property
    def firstline(self):
        return self.lines[0]

    @property
    def numLines(self):
        return len(self.lined)

    @property
    def indices(self):
        return sorted(self.lined.keys())

    @property
    def stanzas(self):
        s2i = {}
        for li in sorted(self.lined.keys()):
            s = tuple(li[1:])
            if s not in s2i:
                s2i[s] = []
            s2i[s] += [li]
        return [l for s, l in sorted(s2i.items())]

    @property
    def stanzas_prosodic(self):
        s2i = {}
        for li in sorted(self.prosodic.keys()):
            s = tuple(li[1:])
            if s not in s2i:
                s2i[s] = []
            s2i[s] += [li]
        return [l for s, l in sorted(s2i.items())]

    @cached_property
    def stanza_length(self):
        """
        Returns invariable stanza length as an integer.
        If variable stanza lengths, returns None.
        """
        stanza_lens = [len(st) for st in self.stanzas]
        if stanza_lens and len(set(stanza_lens)) == 1:
            return stanza_lens[0]
        return None

    @property
    def linenums(self):
        return dict((lineid, lineid[0]) for lineid in self.lined)

    @property
    def stanzanums(self):
        return dict((lineid, lineid[1]) for lineid in self.lined)

    @cached_property
    def linenums_bystanza(self):
        """
        Within stanza numberings
        """
        rd = {}
        stanzanow = None
        linenum = 0
        for lineid, line in sorted(self.lined.items()):
            if lineid[1] != stanzanow:
                stanzanow = lineid[1]
                linenum = 0
            linenum += 1
            rd[lineid] = linenum
        return rd

    def limit(self, N, preserve_stanza=True):
        """Limit number of lines to first N"""
        if preserve_stanza and self.stanza_length:
            lineids = []
            for lineids_in_stanza in self.stanzas:
                lineids += lineids_in_stanza
                if len(lineids) >= N:
                    break
        else:
            lineids = sorted(self.lined.keys())[:N]

        lined = self.lined
        stanza_txts, stanzanow, cur = [], None, []
        for lineid in lineids:
            if lineid[1] != stanzanow:
                if cur:
                    stanza_txts.append('\n'.join(cur))
                stanzanow, cur = lineid[1], []
            cur.append(lined[lineid])
        if cur:
            stanza_txts.append('\n'.join(cur))
        newtxt = '\n\n'.join(stanza_txts)

        id, title, meter, lang = self.id, self.title, self.meter, self.lang
        self.__dict__.clear()
        self.__init__(txt=newtxt, id=id, title=title, meter=meter, lang=lang)

    ## Parsing

    def parse(self, lim=None, meter=None, **meter_kwargs):
        if getattr(self, '_parsed', False) and lim is None and not meter and not meter_kwargs:
            return
        kw = dict(self._meter_kwargs)
        if meter is not None:
            kw.update(self._resolve_meter(meter))
        kw.update(meter_kwargs)
        self.text.parse(lim=lim, **kw)
        self._parsed = True

    @property
    def parsed(self):
        """Dictionary of lineid -> list of best parses (one per line in v3)."""
        self.parse()
        d = {}
        for lineid, line in sorted(self.prosodic.items()):
            bp = _best_parse(line)
            if bp is not None:
                d[lineid] = [bp]
        return d

    @cached_property
    def linelengths(self):
        """Dictionary of lineid -> number of syllables (canonical pronunciation)."""
        df = self.text._syll_df
        canonical = df[(df['form_idx'] == 0) & (~df['is_punc'])]
        counts = canonical.groupby('line_num').size().to_dict()
        return {lineid: int(counts.get(lineid[0], 0)) for lineid in sorted(self.lined)}

    @cached_property
    def linelengths_bybeat(self):
        """Dictionary of lineid -> number of feet (strong positions in best parse)."""
        self.parse()
        d = {}
        for lineid, line in sorted(self.prosodic.items()):
            d[lineid] = num_beats(line)
        return d

    @property
    def linelength(self):  # median line length
        return statistics.median(self.linelengths.values())

    @cached_property
    def numparses(self):
        """Dictionary of lineid -> number of unbounded parses for the line."""
        self.parse()
        d = {}
        for lineid, line in sorted(self.prosodic.items()):
            try:
                d[lineid] = len(line.parses.unbounded)
            except Exception:
                d[lineid] = 0
        return d

    ## Meter

    @property
    def meterd(self):
        """
        Return dictionary with metrical annotations.
        """
        self.parse()
        from prosodic.analysis import classify_meter_type
        mt = classify_meter_type(self.text)

        d = {}
        for k, v in mt['mpos_freqs'].items():
            d['mpos_' + k] = v
        for k, v in mt['perc_lines_ending'].items():
            d['perc_lines_ending_' + k] = v
        for k, v in mt['perc_lines_starting'].items():
            d['perc_lines_starting_' + k] = v
        for k, v in mt['perc_lines_fourth'].items():
            d['perc_lines_fourthpos_' + k] = v
        d['type_foot'] = mt['foot']
        d['type_head'] = mt['head']
        d['type_scheme'] = mt['type']
        d['ambiguity'] = mt['ambiguity'] if mt['ambiguity'] is not None else ''

        # Parse lengths + per-constraint violation rates from the best parses.
        # Constraint names follow prosodic v3 (e.g. 'w_stress'), not v1
        # ('stress_w=>-p'); values are the fraction of positions violating.
        # v3 positions only record nonzero violations, so tally violating
        # positions against the total position count.
        parselens = []
        viol_positions = Counter()
        num_positions = 0
        for lineid, line in sorted(self.prosodic.items()):
            bp = _best_parse(line)
            if bp is None:
                continue
            parselens.append(len(bp.meter_str))
            for pos in bp.positions:
                num_positions += 1
                for cname, cviol in pos.viold.items():
                    if cviol:
                        viol_positions[cname] += 1
        d['length_avg_line'] = statistics.mean(parselens) if parselens else ''
        d['length_avg_parse'] = statistics.mean(parselens) if parselens else ''

        sumviol = 0
        for cname, count in viol_positions.items():
            avg = count / num_positions if num_positions else 0
            d['constraint_' + str(cname).replace('.', '_')] = avg
            sumviol += avg
        d['constraint_TOTAL'] = sumviol
        return d

    @property
    def hood_dist(self):
        d = self.meterd
        try:
            y2 = float(d.get('perc_lines_fourthpos_s', 0))
            x2 = float(d.get('mpos_ww', 0))
        except ValueError:
            return None
        return math.sqrt((0.175 - x2) ** 2 + (0.5 - y2) ** 2)

    @property
    def total_viols(self):
        return self.statd['meter_constraint_TOTAL']

    ## Schemes (line lengths in feet or syllables)

    def get_scheme(self, beat=True, return_diff=False, encourage_invariable=True):
        from prosodic.analysis import detect_line_scheme
        if beat:
            lengths = [v for k, v in sorted(self.linelengths_bybeat.items()) if v is not None]
        else:
            lengths = [v for k, v in sorted(self.linelengths.items())]
        combo, diff = detect_line_scheme(
            lengths,
            beat=beat,
            stanza_length=self.stanza_length,
            encourage_invariable=encourage_invariable,
        )
        if return_diff:
            return combo, diff
        return combo

    @property
    def scheme(self):
        return self.get_scheme(beat=True)

    def get_schemed(self, beat=True):
        scheme, sdiff = self.get_scheme(beat=beat, return_diff=True)
        dx = {}
        dx['scheme'] = scheme
        dx['scheme_type'] = self.schemetype(scheme) if scheme else ''
        dx['scheme_repr'] = self.scheme_repr(dx['scheme_type'], scheme, beat=beat) if scheme else ''
        dx['scheme_length'] = len(scheme) if scheme else 0
        dx['scheme_diff'] = sdiff
        return dx

    @property
    def schemed(self):
        return self.get_schemed(beat=True)

    @property
    def schemed_syll(self):
        return self.get_schemed(beat=False)

    @property
    def schemed_beat(self):
        return self.get_schemed(beat=True)

    def schemetype(self, scheme):
        if len(scheme) == 1:
            return 'Invariable'
        if len(scheme) == 2:
            return 'Alternating'
        return 'Complex'

    def scheme_repr(self, schemetype, scheme, beat=False):
        if beat and schemetype != 'Complex':
            scheme = [BEATNAMES.get(sx, sx) for sx in scheme]
        if schemetype == 'Invariable':
            return scheme[0]
        schemedetails = '(' + '-'.join(str(sx) for sx in scheme) + ')'
        return schemetype + ' ' + schemedetails.lower()

    ## Rhyme

    @cached_property
    def rime_ids(self):
        """Per-line integer rhyme-group ids (0 = unrhymed)."""
        return list(self.text.rhyme_ids)

    @cached_property
    def rhyme_ids(self):
        """Per-line rhyme letters ('-' = unrhymed)."""
        from prosodic.analysis import nums_to_scheme
        return nums_to_scheme(self.rime_ids)

    @property
    def rhymes(self):
        """Dictionary of lineid -> rhyme letter."""
        rd = {}
        for i, lineid in enumerate(sorted(self.lined)):
            try:
                rd[lineid] = self.rhyme_ids[i]
            except IndexError:
                rd[lineid] = '?'
        return rd

    @cached_property
    def rhymed(self):
        odx = {
            'rhyme_scheme': '',
            'rhyme_scheme_name': '',
            'rhyme_scheme_form': '',
            'rhyme_scheme_accuracy': '',
            'rhyme_schemes': None,
        }
        from prosodic.analysis import match_rhyme_scheme
        m = match_rhyme_scheme(self.rime_ids)
        if not m:
            odx['rhyme_scheme'] = 'Unknown'
            return odx
        odx['rhyme_scheme'] = (m['name'], m['form'])
        odx['rhyme_scheme_name'] = m['name']
        odx['rhyme_scheme_form'] = m['form']
        odx['rhyme_scheme_accuracy'] = m['accuracy']
        odx['rhyme_schemes'] = [((n, f), s) for n, f, s in m['candidates']]
        return odx

    def rhyme_net(self, toprint=False, force=False):
        raise NotImplementedError(
            "The networkx rhyme graph was a poesy<=0.3 internal and is gone. "
            "Use poem.rime_ids / poem.rhyme_ids / poem.rhymed instead, or "
            "prosodic.analysis.rhyme_scheme.compute_rhyme_ids for the raw ids."
        )

    ## Statistics and summary

    @property
    def statd(self):
        if not getattr(self, '_statd', None):
            dx = self._statd = {}

            ## Scheme
            for x, y in [('beat', True), ('syll', False)]:
                sd = self.get_schemed(beat=y)
                for sk, sv in sd.items():
                    dx[x + '_' + sk] = sv

            ## Length
            dx['num_lines'] = self.numLines

            ## Meter
            for k, v in self.meterd.items():
                dx['meter_' + k] = v

            ## Rhyme
            for k, v in self.rhymed.items():
                if k == 'rhyme_schemes' and v:
                    v = v[-5:]
                dx[k] = v
        return self._statd

    @property
    def lineld(self):
        old = []
        self.parse()
        for lineid in sorted(self.lined):
            line = self.prosodic[lineid]
            odx = {
                'lineid': lineid,
                '#line': lineid[0],
                '#stanza': lineid[1],
                '#line_in_stanza': self.linenums_bystanza[lineid],
                'lineid2': '%s.%s' % (lineid[1], self.linenums_bystanza[lineid]),

                'line': self.lined[lineid],
                'parse': parse_str(line, viols=True),
                'num_parses': self.numparses[lineid],

                'num_sylls': self.linelengths[lineid],
                'num_feet': self.linelengths_bybeat[lineid],

                'rhyme': self.rhymes[lineid],
            }
            old += [odx]
        return old

    def show(self):
        """Show annotations
        """
        ostr = []
        self.parse()
        stanzanow = None

        lineid2linestr = dict(
            (lineid, parse_str(line, viols=False))
            for lineid, line in sorted(self.prosodic.items())
        )
        maxlinelen = max(len(l) for l in lineid2linestr.values())
        linelen = maxlinelen + 5

        for lineid, linestr in sorted(lineid2linestr.items()):
            rimestr = self.rhymes[lineid]
            linenum = self.linenums_bystanza[lineid]
            stanzanum = lineid[1]
            beatlen = self.linelengths_bybeat[lineid]
            sylllen = self.linelengths[lineid]
            if stanzanow != stanzanum:
                if ostr:
                    ostr += ['']
                stanzanow = stanzanum

            oline = '({stanza}.{linenum}) {line:<{linelen}} [{rime}] [{beat}/{syll}]'.format(
                linelen=linelen,
                rime=rimestr,
                line=linestr,
                stanza=stanzanum,
                linenum=linenum,
                beat=beatlen,
                syll=sylllen)
            ostr += [oline]
        return '\n'.join(ostr)

    def summary(self, header=['lineid2', 'parse', 'rhyme', 'num_feet', 'num_sylls', 'num_parses']):
        colnames = {
            'num_feet': '#feet',
            'num_sylls': '#syll',
            'lineid': '(#ln,#st)',
            'lineid2': '(#s,#l)',
            'rhyme': 'rhyme',
            'parse': 'parse',
            'num_parses': '#parse'
        }

        from tabulate import tabulate
        data = []
        stanzanow = None
        for row in self.lineld:
            if stanzanow is None:
                stanzanow = row['#stanza']
            if stanzanow != row['#stanza']:
                data += ['']
                stanzanow = row['#stanza']

            datarow = [row[h] for h in header]
            data += [datarow]
        cols = [colnames.get(h, h) for h in header]
        table = tabulate(data, headers=cols)

        schemestr1 = 'meter: {meter}\nfeet: {feet}\nsyllables: {syll}\nrhyme: {rhymename} {rhymescheme}'.format(
            meter=str(self.statd['meter_type_scheme']).title(),
            feet=self.statd['beat_scheme_repr'],
            syll=self.statd['syll_scheme_repr'],
            rhymename=self.statd['rhyme_scheme_name'],
            rhymescheme='(%s)' % self.statd['rhyme_scheme_form'] if self.statd['rhyme_scheme_form'] else '',
        )

        ostr = table + '\n\n\nestimated schema\n----------\n' + schemestr1
        print(ostr)

    ## Forms

    @property
    def isSonnet(self):
        return self.text.is_sonnet

    @property
    def isShakespeareanSonnet(self):
        return self.text.is_shakespearean_sonnet

    def __str__(self):
        """
        Return a string version of poem: its ID
        """
        return self.id


def _best_parse(line):
    try:
        return line.best_parse
    except Exception:
        return None


def parse_str(line, viols=True):
    """Render a line's best parse in poesy's classic format,
    e.g. 'WHEN|in.the|CHRON|i|CLE*|of|WAST|ed|TIME'."""
    bp = _best_parse(line)
    if bp is None:
        return ''
    out = []
    for pos in bp.positions:
        s = pos.txt
        if viols and pos.violset:
            s += '*'
        out.append(s)
    return '|'.join(out)


def num_beats(line):
    bp = _best_parse(line)
    return bp.num_peaks if bp is not None else None


#####
# Legacy helper functions (kept for import compatibility)
#####

def transpose(slice):
    unique_numbers = set(slice)
    unique_numbers_ordered = sorted(unique_numbers)
    for i, number in enumerate(slice):
        if number == 0:
            continue
        slice[i] = unique_numbers_ordered.index(number) + 1
    return slice


def scheme2nums(scheme):
    scheme = scheme.replace(' ', '')
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    return [alphabet.index(letter) + 1 if scheme.count(letter) > 1 else 0 for letter in scheme]


def nums2scheme(nums):
    alphabet = '-abcdefghijklmnopqrstuvwxyz'
    return [alphabet[n] if n < len(alphabet) else n for n in nums]


def transpose_up(slice):
    import string
    return ''.join(string.ascii_lowercase[sx - 1] if sx else 'x' for sx in slice)


def schemenums2dict(scheme):
    d = {}
    for i, x in enumerate(scheme):
        for ii, xx in enumerate(scheme[:i]):
            if x == xx:
                d[i] = ii
    return d


def product(*args):
    if not args:
        return iter(((),))  # yield tuple()
    return (items + (item,)
            for items in product(*args[:-1]) for item in args[-1])


def read_tsv(fn, sep='\t'):
    import csv
    with open(fn, encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter=sep))


def hash(string):
    import hashlib
    if type(string) == str:
        string = string.encode('utf-8')
    return str(hashlib.sha224(string).hexdigest())


def toks2freq(l, tfy=False):
    c = Counter(l)
    if tfy:
        summ = float(sum(c.values()))
        for k, v in c.items():
            c[k] = v / summ
    return c


def slicex(l, num_slices=None, slice_length=None, runts=True, random=False):
    """
    Returns a new list of n evenly-sized segments of the original list
    """
    if random:
        import random
        random.shuffle(l)
    if not num_slices and not slice_length:
        return l
    if not slice_length:
        slice_length = int(len(l) / num_slices)
    newlist = [l[i:i + slice_length] for i in range(0, len(l), slice_length)]
    if runts:
        return newlist
    return [lx for lx in newlist if len(lx) == slice_length]


## Constants

BEATNAMES = {
    1: 'Monometer', 2: 'Dimeter', 3: 'Trimeter', 4: 'Tetrameter',
    5: 'Pentameter', 6: 'Hexameter', 7: 'Heptameter', 8: 'Octameter',
    9: 'Enneameter', 10: 'Decameter', 11: 'Hendecameter', 12: 'Dodecameter'}
