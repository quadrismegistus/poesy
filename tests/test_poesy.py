"""Tests for poesy 0.4 (the prosodic-v3 compatibility shim).

Exercises the documented Poem API surface: construction, per-line
dictionaries, parsing, meter/scheme/rhyme statistics, summary, and
sonnet detection.
"""
import io
import os
import sys
from contextlib import redirect_stdout

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from poesy import Poem  # noqa: E402

SONNET_106 = """When in the chronicle of wasted time
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
Had eyes to wonder, but lack tongues to praise."""


@pytest.fixture(scope="module")
def sonnet():
    poem = Poem(SONNET_106)
    poem.parse()
    return poem


# ---------------------------------------------------------------- construction


def test_init_requires_text():
    with pytest.raises(ValueError):
        Poem()


def test_init_from_file(tmp_path):
    fn = tmp_path / "poem.txt"
    fn.write_text(SONNET_106, encoding="utf-8")
    poem = Poem(fn=str(fn))
    assert poem.numLines == 14


def test_basic_attributes(sonnet):
    assert sonnet.numLines == 14
    assert sonnet.title == "When in the chronicle of wasted time"
    assert sonnet.firstline == "When in the chronicle of wasted time"
    assert len(sonnet.lines) == 14
    assert isinstance(sonnet.id, str)


# ---------------------------------------------------------------- line dicts


def test_lined_keys_are_lineids(sonnet):
    lineids = sorted(sonnet.lined.keys())
    assert lineids[0] == (1, 1)
    assert lineids[-1] == (14, 1)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in lineids)


def test_prosodic_dict_gives_line_objects(sonnet):
    for lineid, line in sonnet.prosodic.items():
        assert hasattr(line, "best_parse")
        break
    assert set(sonnet.prosodic.keys()) == set(sonnet.lined.keys())


def test_line_numbering_dicts(sonnet):
    assert sonnet.linenums[(1, 1)] == 1
    assert sonnet.stanzanums[(1, 1)] == 1
    assert sonnet.linenums_bystanza[(14, 1)] == 14


def test_linelengths(sonnet):
    lengths = sonnet.linelengths
    assert len(lengths) == 14
    # iambic pentameter: ~10 syllables per line
    assert 9 <= sonnet.linelength <= 11


def test_linelengths_bybeat(sonnet):
    beats = sonnet.linelengths_bybeat
    assert len(beats) == 14
    import statistics
    assert statistics.median(v for v in beats.values() if v is not None) == 5


def test_numparses(sonnet):
    nps = sonnet.numparses
    assert len(nps) == 14
    assert all(n >= 1 for n in nps.values())


# ---------------------------------------------------------------- meter


def test_meterd(sonnet):
    d = sonnet.meterd
    assert d["type_foot"] == "binary"
    assert d["type_head"] == "final"
    assert d["type_scheme"] == "iambic"
    assert 0 <= d["mpos_w"] <= 1
    assert d["constraint_TOTAL"] >= 0
    assert d["ambiguity"] > 0


def test_total_viols(sonnet):
    assert sonnet.total_viols >= 0


def test_hood_dist(sonnet):
    assert sonnet.hood_dist is not None
    assert sonnet.hood_dist >= 0


# ---------------------------------------------------------------- schemes


def test_schemed_beat(sonnet):
    sd = sonnet.schemed_beat
    assert sd["scheme"] == (5,)
    assert sd["scheme_type"] == "Invariable"
    assert sd["scheme_repr"] == "Pentameter"
    assert sd["scheme_length"] == 1


def test_schemed_syll(sonnet):
    sd = sonnet.schemed_syll
    assert sd["scheme"] == (10,)
    assert sd["scheme_repr"] == 10


def test_scheme_property(sonnet):
    assert sonnet.scheme == (5,)


# ---------------------------------------------------------------- rhyme


def test_rime_ids(sonnet):
    ids = sonnet.rime_ids
    assert len(ids) == 14
    assert any(i > 0 for i in ids)


def test_rhymes_dict(sonnet):
    rd = sonnet.rhymes
    assert len(rd) == 14
    assert all(isinstance(v, str) for v in rd.values())


def test_rhymed(sonnet):
    rd = sonnet.rhymed
    assert "sonnet" in rd["rhyme_scheme_name"].lower()
    assert rd["rhyme_scheme_form"]
    assert rd["rhyme_scheme_accuracy"] > 0
    assert len(rd["rhyme_schemes"]) == 5


def test_rhyme_net_raises(sonnet):
    with pytest.raises(NotImplementedError):
        sonnet.rhyme_net()


# ---------------------------------------------------------------- stats & summary


def test_statd(sonnet):
    d = sonnet.statd
    assert d["num_lines"] == 14
    assert d["meter_type_scheme"] == "iambic"
    assert d["beat_scheme_repr"] == "Pentameter"
    assert d["syll_scheme_repr"] == 10
    assert "sonnet" in d["rhyme_scheme_name"].lower()


def test_lineld(sonnet):
    rows = sonnet.lineld
    assert len(rows) == 14
    row = rows[0]
    for key in ("lineid", "#line", "#stanza", "lineid2", "line", "parse",
                "num_parses", "num_sylls", "num_feet", "rhyme"):
        assert key in row
    assert row["lineid2"] == "1.1"
    assert "|" in row["parse"]


def test_summary_prints_table(sonnet):
    buf = io.StringIO()
    with redirect_stdout(buf):
        sonnet.summary()
    out = buf.getvalue()
    assert "estimated schema" in out
    assert "meter: Iambic" in out
    assert "feet: Pentameter" in out
    assert "rhyme:" in out


def test_show(sonnet):
    out = sonnet.show()
    assert out.count("\n") >= 13
    assert "(1.1)" in out


# ---------------------------------------------------------------- forms


def test_is_sonnet(sonnet):
    assert sonnet.isSonnet is True


def test_shakespearean_in_candidates(sonnet):
    # Sonnet 106 is slant-rhyme-heavy, so the *best* match may be a sonnet
    # variant, but the Shakespearean form should rank among the candidates.
    names = [name for (name, form), score in sonnet.rhymed["rhyme_schemes"]]
    assert any("Shakespearean" in n for n in names)


def test_is_shakespearean_sonnet():
    fn = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                      "poems", "shakespeare_sonnets", "sonnet-018.txt")
    poem = Poem(fn=fn)
    assert poem.isShakespeareanSonnet is True


def test_non_sonnet():
    poem = Poem("The cat\nsat on the mat.\nThe dog\nlay on the log.")
    assert poem.isSonnet is False


# ---------------------------------------------------------------- misc API


def test_stanzas(sonnet):
    assert len(sonnet.stanzas) == 1
    assert len(sonnet.stanzas[0]) == 14
    assert sonnet.stanza_length == 14


def test_multistanza():
    poem = Poem("The cat\nsat on the mat.\n\nThe dog\nlay on the log.")
    assert len(poem.stanzas) == 2
    assert poem.stanza_length == 2
    lineids = sorted(poem.lined.keys())
    assert lineids == [(1, 1), (2, 1), (3, 2), (4, 2)]


def test_limit():
    poem = Poem("The cat\nsat on the mat.\n\nThe dog\nlay on the log.")
    poem.limit(2)
    assert poem.numLines == 2
    assert sorted(poem.lined.keys()) == [(1, 1), (2, 1)]


def test_legacy_meter_name_warns():
    with pytest.warns(UserWarning):
        Poem("The cat\nsat on the mat.", meter="iambic_pentameter")


def test_default_meter_name_silent(recwarn):
    Poem("The cat\nsat on the mat.", meter="default_english")
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
