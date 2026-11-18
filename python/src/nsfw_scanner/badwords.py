"""DNS-level badword scanner.

Some hostnames are unambiguous: ``pornhub.com``, ``xxx-videos.net``,
``onlyfans.com``, ``hentai.example``.  Downloading the page (and running
the ONNX models) to discover that is wasted hardware.  This module rates a
hostname from its *name only*:

    10  a strong/explicit term matched
    7   only suggestive terms matched
    None  nothing conclusive (the caller should scan normally)

Matching is deliberately conservative:

* the host is split into labels/tokens (``.``, ``-``, ``_``) and compared
  as whole tokens for short/ambiguous terms (``sex``, ``xxx``, ``anal``,
  ``cum``, ``ass``);
* longer unambiguous terms may match as substrings, so ``pornhub`` and
  ``freeporn`` both fire;
* a blocklist of innocent look-alikes (``essex``, ``sussex``,
  ``analysis``, ``dickens``, ``peacock``, ``grape``, ...) is masked before
  substring matching, so ``essex.gov.uk`` and ``analysis.com`` stay safe.

Public helpers:
    domain_badwords(target) -> dict
    badword_rating(target)  -> 10 | 7 | None
    is_bad_domain(target, threshold=10) -> bool
"""

import re
from urllib.parse import urlparse

# Strong = explicit on its own.  Rating 10.
# Short/ambiguous terms are token-only (see _STRONG_TOKENS); longer ones are
# matched as substrings (see _STRONG_SUBSTRINGS).
_STRONG_SUBSTRINGS = (
    "porn", "porno", "pornography", "pornstar",
    "hentai", "nsfw", "nudity", "nude", "naked",
    "erotic", "fetish", "bdsm", "bondage", "dominatrix",
    "camgirl", "camsex", "sexcam", "sextape", "sexvideo", "sexchat",
    "hardcore", "milf", "incest", "escort", "onlyfans",
    "blowjob", "handjob", "cumshot", "creampie", "bukkake", "gangbang",
    "dildo", "masturbat", "orgasm", "orgies", "threesome", "swinger",
    "gloryhole", "upskirt", "downblouse", "voyeur", "sadomasoch",
    "fuck", "slut", "whore", "pussy", "titjob", "boobs",
    "lolicon", "shotacon", "childporn",
    # Hebrew (no word boundaries in DNS labels; substring is intentional).
    "סקס", "פורנו", "פורנוגרפיה", "עירום", "עירומה", "עירומים", "עירומות",
    "ארוטיקה", "זונה", "זונות", "חשפנות", "חשפן", "חשפנית",
    "מציצה", "שדיים", "פין", "נרתיק", "אורגזמה", "מאסטורבציה",
)

_STRONG_TOKENS = {
    # Ambiguous as substrings (Essex, sextant, analysis, document, class,
    # title, xxxl...), explicit as a whole domain label/token.
    "sex", "xxx", "anal", "cum", "ass", "tit", "cock", "dick", "rape",
    "rapist", "adultvideo", "adultcam", "adultcontent",
}

# Suggestive only.  Rating 7.
_WEAK_SUBSTRINGS = (
    "sensual", "lingerie", "bikini", "striptease", "seduction",
    "kinky", "hookup", "adult",
)
_WEAK_TOKENS = {
    "sexy", "adult", "striptease", "erotica",
}
# Hebrew suggestive terms.
_WEAK_HEBREW = ("מין", "ליווי", "חשוף", "מפתה", "יצרים")

# Innocent words that contain a strong term as a substring.  Masked before
# substring matching so the classic Scunthorpe problem does not fire.
_SAFE_MASK = (
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
    "adulthood",
)
_SAFE_MASK = tuple(sorted(_SAFE_MASK, key=len, reverse=True))

_TOKEN_SPLIT = re.compile(r"[.\-_\s]+")
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def _host_of(target):
    """Extracts the lowercase hostname from a URL or bare domain."""
    text = (target or "").strip()
    if not text:
        return ""
    if not _SCHEME_RE.match(text):
        text = "//" + text.lstrip("/")
    try:
        parsed = urlparse(text)
        host = parsed.hostname or ""
    except ValueError:
        host = text.split("/", 1)[0]
    host = host.strip().lower().rstrip(".")
    # Drop userinfo / port leftovers if urlparse could not parse it.
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host and not host.startswith("["):
        host = host.rsplit(":", 1)[0]
    return host


def _tokens(host):
    return [tok for tok in _TOKEN_SPLIT.split(host) if tok]


def _masked_flat(host):
    flat = host.replace(".", "").replace("-", "").replace("_", "")
    for word in _SAFE_MASK:
        if word in flat:
            flat = flat.replace(word, "*" * len(word))
    return flat


def domain_badwords(target):
    """Rates a hostname from its name alone.

    Returns a dict::

        {"host", "matches", "strong", "weak", "rating", "scanner", "reason"}

    ``rating`` is 10 for a strong match, 7 for suggestive-only, else None.
    """
    host = _host_of(target)
    out = {"host": host, "matches": [], "strong": [], "weak": [],
           "rating": None, "scanner": "dns-badwords", "reason": "no-match"}
    if not host:
        out["reason"] = "empty-host"
        return out

    tokens = _tokens(host)
    token_set = set(tokens)
    flat = _masked_flat(host)

    strong = []
    for term in _STRONG_SUBSTRINGS:
        if term in flat:
            strong.append(term)
    for term in _STRONG_TOKENS:
        if term in token_set:
            strong.append(term)
    # Hebrew weak terms are unambiguous enough as substrings too.
    weak = []
    for term in _WEAK_SUBSTRINGS:
        if term in flat:
            weak.append(term)
    for term in _WEAK_TOKENS:
        if term in token_set:
            weak.append(term)
    for term in _WEAK_HEBREW:
        if term in host:
            weak.append(term)

    # Never report the same surface twice (strong wins).
    weak = [w for w in weak if w not in strong]
    out["strong"] = sorted(set(strong))
    out["weak"] = sorted(set(weak))
    out["matches"] = out["strong"] + out["weak"]

    if out["strong"]:
        out["rating"] = 10
        out["reason"] = "strong-badword"
    elif out["weak"]:
        out["rating"] = 7
        out["reason"] = "suggestive-badword"
    return out


def badword_rating(target):
    """Convenience: 10 / 7 / None for a hostname."""
    return domain_badwords(target)["rating"]


def is_bad_domain(target, threshold=10):
    """True when the hostname rating is at least ``threshold``."""
    rating = badword_rating(target)
    return rating is not None and rating >= threshold
