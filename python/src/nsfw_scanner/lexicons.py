"""Configurable sensuality lexicon.

The terms that decide a text verdict live in plain text files so they can
be extended without touching code::

    packages/nsfw-lib/src/nsfw_scanner/data/lexicons/sensuality.txt
    packages/nsfw-lib/src/nsfw_scanner/data/lexicons/sensuality_safe.txt

Line syntax (``sensuality.txt``):

    porn              # word-boundary match (default) - "sex" never fires
                      # on "Essex", "porn" does not fire on "Capricorn"
    ~pornhub          # substring match - only for long, unambiguous terms
    -cams             # drop a built-in term
    # anything after a '#' is a comment

``sensuality_safe.txt`` lists look-alike words that are masked *before*
substring matching, so the Scunthorpe problem does not fire
("analysis" contains "anal", "essex" contains "sex", "peacock" contains
"cock", ...).

Point ``NSFW_LEXICON_DIR`` at your own directory to use your own files;
the built-in terms are always kept unless a file removes them with
``-term``.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEXICON_DIR = os.path.join(HERE, "data", "lexicons")

# The original sensuality list (unchanged since the first commit) plus the
# additions below.  Order matters only for the debug output.
BUILTIN_TERMS = (
    # --- original list -------------------------------------------------
    "porn", "xxx", "sex", "nude", "adult", "erotic", "fetish", "bdsm",
    "cams", "escort", "taboo", "incest", "hentai", "milf", "nsfw",
    "explicit", "hardcore", "strip", "lust",
    # Hebrew adult keywords.
    "סקס", "פורנו", "פורנוגרפיה", "עירום", "עירומה", "עירומים", "עירומות",
    "סקסי", "סקסית", "ארוטיקה", "ארוטי", "ארוטית", "זונה", "זונות",
    "חשפנות", "חשפן", "חשפנית",
    # --- added: unambiguous, single-word sensuality terms ---------------
    "topless", "undressed", "horny", "aroused", "erotica", "ecchi",
    "futanari", "shemale", "orgy", "blowjob", "handjob", "rimjob",
    "titjob", "footjob", "creampie", "cumshot", "deepthroat",
    "softcore", "hardcore", "pornstar", "sextape", "sexvideo", "sexchat",
    "camsex", "camgirl", "adultvideo", "adultcam", "adultcontent",
    "brazzers", "onlyfans", "fansly", "chaturbate", "motherless",
    # Hebrew additions.
    "בילי בייבי", "שדיים", "מיני מקלט", "פורני מקלט",
)

# Terms matched as substrings (long and unambiguous only).  A substring
# match can hit inside a bigger word, so the safe mask is applied first.
BUILTIN_SUBSTRING_TERMS = (
    "pornhub", "xnxx", "xvideos", "xhamster", "redtube", "youporn",
    "tube8", "spankbang", "e621", "rule34", "hentai", "nsfw", "futanari",
    "onlyfans", "fansly", "chaturbate", "brazzers", "shemale",
    "pornstar", "escort", "erotica", "bdsm", "fetish", "incest",
    "pornography", "porno", "sexvideo", "sextape", "sexchat", "camsex",
    "camgirl", "adultvideo", "adultcam", "adultcontent", "topless",
    "סקס", "פורנו", "פורנוגרפיה", "עירום", "זונה", "חשפנות",
)

# Look-alike words masked before substring matching.
BUILTIN_SAFE_MASK = (
    # sex
    "essex", "sussex", "middlesex", "sextant", "sextet", "sexton",
    "scunthorpe",
    # anal
    "analysis", "analyst", "analytics", "analog", "analogue", "canal",
    "banal", "banana",
    # rape
    "grape", "grapefruit", "therapist", "therapy", "therapeutic",
    # dick / cock
    "dickens", "dickinson", "hitchcock", "peacock", "cocktail", "cockpit",
    "gamecock", "woodcock",
    # adult
    "adulthood", "adulting",
    # xxx
    "class", "classic", "classes", "clastic", "clause", "clumsy",
    "document", "documents", "appendix", "excise", "excited", "exciting",
    "excuse", "executive", "execute", "exodus", "textile", "texts",
    "taxi", "taxes", "tuxedo",
)


def _read_terms(path):
    """Reads a lexicon file into (add, substring_add, remove) term lists."""
    add, substr, remove = [], [], []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError:
        return add, substr, remove
    for line in raw.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):
            remove.append(line[1:].strip())
        elif line.startswith("~"):
            term = line[1:].strip()
            if term:
                substr.append(term)
        else:
            add.append(line)
    return add, substr, remove


def _lexicon_dir():
    return os.environ.get("NSFW_LEXICON_DIR", DEFAULT_LEXICON_DIR)


_CACHE = {"key": None, "value": None}


def _lexicon_files(directory):
    return (os.path.join(directory, "sensuality.txt"),
            os.path.join(directory, "sensuality_safe.txt"))


def _lexicon_key(directory):
    """Cache key: the directory + the mtimes of the files in it."""
    key = [directory]
    for path in _lexicon_files(directory):
        try:
            key.append(os.path.getmtime(path))
        except OSError:
            key.append(None)
    return tuple(key)


def load_lexicon():
    """Returns the effective lexicon.

    ``{"terms": [...], "substring_terms": [...], "safe_mask": [...]}``

    The built-in terms are always the base; the files in
    ``NSFW_LEXICON_DIR`` (default: the packaged ``data/lexicons``) add to
    them, and can remove a built-in term with ``-term``.  The result is
    cached until one of the files changes.
    """
    directory = _lexicon_dir()
    key = _lexicon_key(directory)
    if _CACHE["key"] == key:
        return _CACHE["value"]

    terms = list(BUILTIN_TERMS)
    substr = list(BUILTIN_SUBSTRING_TERMS)
    mask = list(BUILTIN_SAFE_MASK)

    terms_file, mask_file = _lexicon_files(directory)
    add, substr_add, remove = _read_terms(terms_file)
    _, _, mask_add = _read_terms(mask_file)

    if remove:
        drop = {t.lower() for t in remove}
        terms = [t for t in terms if t.lower() not in drop]
        substr = [t for t in substr if t.lower() not in drop]
    for term in add:
        if term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
    for term in substr_add:
        if term.lower() not in {t.lower() for t in substr}:
            substr.append(term)
    for word in mask_add:
        if word.lower() not in {w.lower() for w in mask}:
            mask.append(word)
    value = {"terms": terms, "substring_terms": substr, "safe_mask": mask}
    _CACHE["key"] = key
    _CACHE["value"] = value
    return value


def sensuality_lexicon():
    """The effective sensuality lexicon (public, for inspection)."""
    return load_lexicon()
