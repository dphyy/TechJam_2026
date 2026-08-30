from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field

from mercury.types import FeedbackDecision, OverrideDecision, OverrideFact, Preference
from mercury.vocabulary import CatalogVocabulary, VocabularyMatch


@dataclass(slots=True)
class SourceRecord:
    """A session-local source record; conversation text is never written to disk."""

    turn: int
    text: str
    preferences: list[Preference] = field(default_factory=list)
    informative: bool = False


@dataclass(slots=True)
class _Assertion:
    preference: Preference
    additive: bool = False
    replacement: bool = False
    clause: int = 0
    choice_replacement: bool = False


# Canonical values match the catalog facets. Aliases concern ordinary shopping
# language, independent of any evaluation dialogue or product identifier.
_VALUES = {
    "material": {
        "cotton": ("cotton",), "leather": ("leather",),
        "faux leather": ("faux leather", "vegan leather", "imitation leather", "pleather"),
        "polyester": ("polyester",), "nylon": ("nylon",), "wool": ("wool", "woolen", "woollen"),
        "merino wool": ("merino wool", "merino"), "silk": ("silk",), "linen": ("linen",),
        "denim": ("denim",), "suede": ("suede",), "rubber": ("rubber",), "canvas": ("canvas",),
        "mesh": ("mesh",), "fleece": ("fleece",), "spandex": ("spandex", "lycra", "elastane"),
        "rayon": ("rayon", "viscose"), "cashmere": ("cashmere",), "velvet": ("velvet",),
        "satin": ("satin",), "acrylic": ("acrylic",), "bamboo": ("bamboo",),
        "stainless steel": ("stainless steel",), "sterling silver": ("sterling silver",),
        "gold": ("solid gold", "gold plated"), "silicone": ("silicone",),
    },
    "color": {
        **{value: (value,) for value in (
            "black", "white", "blue", "navy", "red", "green", "yellow", "pink", "purple",
            "brown", "beige", "orange", "silver", "gold", "burgundy", "khaki", "cream",
            "tan", "teal", "maroon", "olive", "ivory", "coral", "turquoise", "lavender",
        )},
        "grey": ("grey", "gray"), "multicolor": ("multicolor", "multicolour", "multicolored"),
    },
    "category": {
        "shoes": ("shoe", "shoes", "footwear"), "boots": ("boot", "boots"),
        "sandals": ("sandal", "sandals"),
        "sneakers": ("sneaker", "sneakers", "trainer", "trainers", "running shoes", "running shoe"),
        "slippers": ("slipper", "slippers"), "flats": ("flats",), "heels": ("heels", "high heels"),
        "shirts": ("shirt", "shirts", "blouse", "blouses"),
        "t-shirts": ("t-shirt", "t-shirts", "t shirt", "t shirts", "tee", "tees"),
        "tops": ("top", "tops", "tank top", "tank tops"),
        "pants": ("pants", "trousers", "trouser"), "jeans": ("jeans",), "shorts": ("shorts",),
        "leggings": ("leggings", "tights"), "skirts": ("skirt", "skirts"),
        "dresses": ("dress", "dresses", "gown", "gowns"),
        "jackets": ("jacket", "jackets", "blazer", "blazers"), "coats": ("coat", "coats"),
        "sweaters": ("sweater", "sweaters", "jumper", "jumpers", "pullover", "cardigan"),
        "hoodies": ("hoodie", "hoodies", "sweatshirt", "sweatshirts"),
        "socks": ("sock", "socks"), "underwear": ("underwear", "briefs", "boxers"),
        "bras": ("bra", "bras"), "swimwear": ("swimwear", "swimsuit", "bikini", "swimming costume"),
        "hats": ("hat", "hats", "cap", "caps", "beanie"), "gloves": ("glove", "gloves", "mittens"),
        "scarves": ("scarf", "scarves"), "belts": ("belt", "belts"),
        "bags": ("bag", "bags", "handbag", "handbags", "purse", "purses", "tote"),
        "backpacks": ("backpack", "backpacks", "rucksack"), "wallets": ("wallet", "wallets"),
        "watches": ("watch", "watches"), "rings": ("ring", "rings"),
        "necklaces": ("necklace", "necklaces", "pendant", "pendants"),
        "earrings": ("earring", "earrings"), "bracelets": ("bracelet", "bracelets", "bangle", "bangles"),
        "jewelry": ("jewelry", "jewellery"), "sunglasses": ("sunglasses",),
    },
    "style": {
        **{value: (value,) for value in (
            "casual", "formal", "vintage", "classic", "elegant", "minimalist", "floral",
            "striped", "plaid", "oversized", "cropped", "polka dot", "embroidered",
        )},
        "athletic": ("athletic", "sporty"), "bohemian": ("bohemian", "boho"),
        "slim fit": ("slim fit", "slim-fit", "fitted"),
        "relaxed fit": ("relaxed fit", "relaxed-fit"), "loose fit": ("loose fit", "loose-fit"),
    },
    "use_case": {
        "running": ("running", "jogging", "jog"), "walking": ("walking", "walks"),
        "hiking": ("hiking", "trekking", "hike"), "swimming": ("swimming", "swim"),
        "wedding": ("wedding", "weddings", "bridal"), "work": ("work", "office", "workplace"),
        "travel": ("travel", "traveling", "travelling"), "yoga": ("yoga",),
        "cycling": ("cycling", "biking"), "gym": ("gym", "workout", "workouts"),
        "winter": ("winter", "cold weather"), "summer": ("summer", "hot weather"),
        "outdoor": ("outdoor", "outdoors"), "beach": ("beach",), "party": ("party", "parties"),
        "everyday": ("everyday", "daily wear"), "sleep": ("sleep", "sleeping", "bedtime"),
    },
    "feature": {
        "waterproof": ("waterproof", "water proof"), "water resistant": ("water resistant", "water-resistant"),
        "breathable": ("breathable", "breathability"), "lightweight": ("lightweight", "light weight"),
        "adjustable": ("adjustable",), "stretch": ("stretch", "stretchy", "stretchable"),
        "pockets": ("pocket", "pockets"), "zippered pockets": ("zippered pockets", "zipped pockets", "zip pockets"),
        "arch support": ("arch support",), "slip resistant": ("slip resistant", "non-slip", "nonslip"),
        "machine washable": ("machine washable", "machine-washable"),
        "quick dry": ("quick dry", "quick-dry", "quick drying", "quick-drying"),
        "insulated": ("insulated", "insulation"), "padded": ("padded", "padding"),
        "reversible": ("reversible",), "seamless": ("seamless",), "hood": ("hood", "hooded"),
        "zipper": ("zipper", "zippered", "zip closure", "zip up", "zip-up"),
        "long sleeve": ("long sleeve", "long sleeves", "long-sleeved"),
        "short sleeve": ("short sleeve", "short sleeves", "short-sleeved"),
        "no show": ("no show", "no-show"),
        "sleeveless": ("sleeveless",), "wide fit": ("wide fit", "wide width"),
        "comfortable": ("comfortable", "comfort"), "durable": ("durable", "durability"),
        "moisture wicking": ("moisture wicking", "moisture-wicking", "sweat wicking"),
        "hypoallergenic": ("hypoallergenic",), "supportive": ("supportive",),
    },
}
_LOOKUPS = {
    attribute: {alias: canonical for canonical, aliases in values.items() for alias in aliases}
    for attribute, values in _VALUES.items()
}
_COMPONENT_AFTER = re.compile(
    r"^\s+(?:on|for|in|with|as)?\s*(body|lining|handles?|straps?|zipper|sole|laces?|"
    r"sleeves?|collar|cuffs?|pockets?|closure)\b"
)
_PATTERNS = {
    attribute: re.compile(r"(?<!\w)(?:" + "|".join(
        re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)
    ) + r")(?!\w)")
    for attribute, aliases in _LOOKUPS.items()
}
_ATTRIBUTE_WORDS = {
    "category": r"categor(?:y|ies)|type of (?:item|product)|product type",
    "material": r"materials?|fabrics?", "color": r"colou?rs?|shades?", "size": r"sizes?|sizing",
    "style": r"styles?|patterns?", "brand": r"brands?|makers?", "budget": r"budget|prices?|cost|spend",
    "feature": r"features?|details?", "use_case": r"use case|occasion|activity", "other": r"other",
}
_NO_PREFERENCE = re.compile(
    r"\b(?:no (?:particular |strong |specific )?(?:(?:material|fabric|color|colour|size|style|brand) )?preference|"
    r"(?:don't|do not) have (?:a |any )?(?:particular |strong |specific )?"
    r"(?:(?:material|fabric|color|colour|size|style|brand) )?preference|"
    r"don't (?:really )?care|do not care|flexible (?:on|about)|"
    r"doesn't matter|does not matter|anything (?:is |would be )?(?:fine|okay|ok)|"
    r"(?:any|either)\b.*\b(?:fine|okay|ok|works|will do)|not (?:fussy|fussed|picky)|"
    r"no (?:budget|price) (?:limit|cap))\b"
)
_CANONICAL_NO_PREFERENCE = re.compile(
    r"\b(?:no longer (?:have|want) (?:a |any )?(?:particular |strong |specific )?"
    r"(?:(?:material|fabric|color|colour|size|style|brand|feature|category) )?preference|"
    r"(?:(?:material|fabric|color|colour|size|style|brand|feature|category) )?"
    r"(?:is|are) no longer (?:important|required|needed|a preference))\b"
)
_NO_NEW_INFORMATION = re.compile(
    r"\b(?:(?:no|not|don't have|do not have|haven't|have not)(?:\s+(?:a|an|any))?\s+"
    r"(?:new|more|additional|further|extra)(?:\s+[a-z]+){0,2}?\s+"
    r"(?:preferences?|details?|information|requirements?)|"
    r"nothing(?:\s+(?:new|more|additional|further|else))?\s+to\s+(?:add|share|say))\b"
)
_UNPRODUCTIVE = re.compile(
    r"\b(?:not sure|don't know|do not know|unsure|rather not (?:say|answer|share|discuss)|"
    r"prefer not to (?:say|answer|share)|don't want to (?:say|answer|share|discuss)|"
    r"can't (?:say|answer)|cannot (?:say|answer)|not (?:sharing|comfortable (?:sharing|answering|discussing))|"
    r"no (?:new|more|additional) (?:preferences|details|information)|"
    r"(?:nothing|no more|anything else|any more preferences) to add|"
    r"none of (?:these|those)|not (?:these|those)|(?:these|those) (?:aren't|are not|don't|do not) "
    r"(?:right|working|work)|different options|another (?:set|batch)|keep looking|surprise me)\b"
)
_ACTIONABLE_REJECTION = re.compile(
    r"\b(?:not (?:these|those)|none of (?:these|those)(?: (?:is|are) right| (?:work|works|fit|fits)|"
    r" (?:feel|feels) right)?|"
    r"(?:these|those) (?:aren't|are not|don't|do not) "
    r"(?:right|working|work))\b"
)
_NOT_JUST_COMPONENT = re.compile(r"\bnot\s+(?:just|only|merely)\b")
_REPLACEMENT = re.compile(
    r"\b(?:actually|instead|rather than|after all|on second thought|change|switch|replace|"
    r"make that|go with|changed my mind|scratch that|swap(?: it| that)?)\b"
)
_IMPLICIT_PREFERENCE_SHIFT = re.compile(
    r"\b(?:let'?s try|back to|(?:works|sounds|feels|looks|seems) better|"
    r"makes? (?:more|better) sense|not feeling)\b"
)
_EXPLICIT_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget) (?:my |the )?(?:earlier|previous|old)|"
    r"\b(?:on second thought|changed my mind|make that|scratch that|swap(?: it| that)?|"
    r"switch(?: that| it)?|change (?:that|it)|no longer|rather than|let'?s try|back to|"
    r"not feeling|(?:works|sounds|feels|looks|seems) better|makes? (?:more|better) sense)\b"
)
_POSSIBLE_OVERRIDE = re.compile(r"\b(?:actually|instead|after all|go with|change|replace)\b")
_NO_CHANGE_GUARD = re.compile(
    r"\b(?:(?:still|already) (?:is |are |works? )?(?:fine|okay|ok|good)|"
    r"(?:do not|don't) need to change|no (?:need|reason) to change|nothing to change|"
    r"instead of changing\b.{0,48}\bkeep|keep\b.{0,48}\b(?:the )?same)\b"
)
_EITHER_OR_GUARD = re.compile(r"\b(?:either\b.{0,48}\bor|(?:any|either) (?:one|option))\b")
_ATTRIBUTE_CHANGE_REQUEST = re.compile(
    r"\b(?:different|another|new)\s+(material|fabric|colou?r|style|brand|size|category|"
    r"product type|features?)\b|\bexcept\s+(?:the\s+)?(material|fabric|colou?r|style|"
    r"brand|size|category|product type|features?)\b"
)
_ADDITIVE = re.compile(r"\b(?:also|as well|in addition|either|or|too)\b")
_SOFT = re.compile(r"\b(?:prefer|preference|ideally|maybe|perhaps|would be nice|if possible|leaning|nice to have)\b")
_HARD = re.compile(r"\b(?:must|need|needs|required|only|essential|have to|has to|cannot|can't)\b")
_PRODUCT_TYPE_REJECTION = re.compile(
    r"\b(?:not|don't want|do not want) (?:this|that) (?:product )?(?:type|category|kind)\b"
)
_ITEM_REJECTION = re.compile(
    r"\b(?:(?:not|don't want|do not want) (?:this|that|these|those) "
    r"(?:item|items|one|ones|option|options|product|products)|"
    r"none of (?:these|those)(?: (?:is|are) right| (?:work|works|fit|fits)| (?:feel|feels) right)?|"
    r"(?:these|those) (?:aren't|are not|don't|do not) (?:right|working|work))\b"
)
_ATTRIBUTE_REJECTION = re.compile(
    r"\b(?:not|don't want|do not want) (?:this|that) "
    r"(material|fabric|colou?r|style|brand|size|feature)\b"
)
# An approximate budget states a target, not a limit a product must satisfy.
# Hedges are matched near the money span because several budget patterns absorb
# the hedge word themselves, as in "budget around $40".
_BUDGET_HEDGE = re.compile(r"\b(?:around|about|approximately|approx|roughly|circa|or so)\b|~")
_SUFFIX_RELAXATION = re.compile(
    r"\b(?:isn't|is not|aren't|are not|no longer)\s+(?:necessary|required|important|needed)\b"
)
_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_MONEY = r"(?:usd\s*|sgd\s*|us\s*)?\$?\s*(" + _NUMBER + r")"
_UNITS = {"%": "%", "percent": "%", "mm": "mm", "millimeter": "mm", "millimeters": "mm",
          "millimetre": "mm", "millimetres": "mm", "cm": "cm", "centimeter": "cm",
          "centimeters": "cm", "centimetre": "cm", "centimetres": "cm", "in": "inch",
          "inch": "inch", "inches": "inch", "oz": "oz", "ounce": "oz", "ounces": "oz",
          "g": "g", "gram": "g", "grams": "g", "kg": "kg", "lb": "lb", "lbs": "lb"}
_UNIT_PATTERN = r"(?:%|" + "|".join(re.escape(unit) + r"\b" for unit in sorted(_UNITS, key=len, reverse=True) if unit != "%") + r")"
_QUANTITY_PATTERN = re.compile(r"(?<![\w.])(" + _NUMBER + r")\s*(" + _UNIT_PATTERN + r")?\s*[- ]?\s*")
_SIZE_ALIASES = {"extra small": "xs", "small": "s", "medium": "m", "large": "l", "extra large": "xl",
                 "extra extra large": "xxl", "xx-small": "xxs", "x-small": "xs", "x-large": "xl", "xx-large": "xxl"}
_SIZE_VALUE = r"(?:extra extra large|extra (?:small|large)|xx?-small|xx?-large|small|medium|large|xxxl|xxl|xxs|xs|xl|[sml]|\d+(?:\.\d+)?)"
_RESIDUAL_STOPWORDS = frozenset("""
    a an the this that these those it its it's they're their i i'm i'd i'll i've me my mine
    we we're our us you your yours he she his her they them one ones some any anything
    something nothing everything none either neither all each both another other else
    am is isn't are aren't was were be been being do don't does doesn't did didn't
    have has had having can can't cannot could would should will won't must may might
    to of in on at for from by with without into about as than and or nor but except so if
    because while when where which who how what whether not no yes yeah okay ok sure please
    thanks thank just really very quite maybe perhaps ideally actually also too well
    want wants wanted need needs needed like liked prefer preferred preference preferences
    looking look find searching search shop shopping buy buying browse browsing after
    get give show see pick choose keep continue try trying use wear wearing made make go
    meant mean thought second instead rather change changed switch replace add answer say tell
    know discuss share provide more less most much many few only still yet already now
    then again ever always never possible fine nice good great right wrong better best longer
    different same previous next options option choices choice recommendations recommendation
    item items product products thing things details detail information new particular
    specific strong necessary required important anymore maximum minimum budget dollars
    dollar price cost spend stretch cap capped pair around approximately roughly under
    over above below between least most enough comfortable happy willing interested
    brand material fabric color colour style size feature features category occasion
    avoid excluding exclude hate dislike ask asked asking attribute attributes
    earlier ignore ignoring explore exploring requirement requirements require another
    matter matters mind scratch swap judgment judgement
""".split())


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'").replace("–", "-").replace("—", " ")).strip()


def _mentioned_attributes(text: str) -> set[str]:
    return {attribute for attribute, pattern in _ATTRIBUTE_WORDS.items() if re.search(r"\b(?:" + pattern + r")\b", text)}


def _polarity(clause: str, start: int, end: int) -> int:
    prefix = clause[:start]
    suffix = clause[end:]
    if re.search(r"\bnot feeling (?:the )?$", prefix):
        return -1
    if re.search(r"\b(?:any(?:\s+\w+){0,2}|anything|everything)\s+other\s+than\s*$", prefix):
        return -1
    if re.search(r"\bexcept\s*$", prefix):
        return -1
    # The last contrast/replacement boundary defines the negation's scope.
    prefix = re.split(r"\b(?:but|however|except)\b", prefix)[-1]
    if re.search(r"\b(?:instead of|rather than)\s*(?:the\s+)?$", prefix):
        return -1
    if re.search(
        r"(?:^|\b(?:actually|please|and|but)\s+)(?:scratch|ditch|drop|remove|skip|"
        r"forget(?: about)?|leave out)\s+(?:the\s+)?$",
        prefix,
    ):
        return -1
    if re.search(r"\b(?:is|are|feel|feels|seem|seems|look|looks)\s+too\s+$", prefix):
        return -1
    if re.match(r"(?:[- ]free\b|\s+(?:is|are)\s+(?:not (?:okay|ok|fine|wanted)|out)\b)", suffix):
        return -1
    prefix = re.sub(r"\b(?:no|not) (?:more|less) than\b", "", prefix)
    # A no-show design is a compound descriptor, not a refusal of the item.
    prefix = re.sub(r"\bno[- ]+show\b", "", prefix)
    # "No matter if/whether" introduces a concessive description; its "no"
    # does not negate the following product terms.
    prefix = re.sub(r"\bno\s+matter\b(?:\s+(?:if|whether))?", "", prefix)
    if re.search(r"\b(?:no|not|without|avoid|excluding|exclude|hate|dislike|don't want|do not want)\b", prefix):
        if not re.search(r"\bnot (?:only|just|merely)\b", prefix):
            return -1
    return 1


def _number(value: str) -> str:
    return format(float(value.replace(",", "")), "g")


def _budget(clause: str, prompted: bool) -> list[tuple[str, int, int]]:
    named_money = bool(re.search(r"\$|\b(?:dollars?|budget|price|cost|spend|afford)\b", clause))

    def monetary_tail(match: re.Match[str]) -> bool:
        tail = clause[match.end():].strip()
        return not re.match(_UNIT_PATTERN, tail) and (
            named_money or not tail or tail == "please" or bool(re.match(r"(?:please\s+)?for\b", tail))
        )

    match = re.search(r"\bbetween\s+" + _MONEY + r"\s+and\s+" + _MONEY, clause)
    if not match:
        match = re.search(_MONEY + r"\s*(?:-|to)\s*" + _MONEY, clause)
    money_context = prompted or named_money
    if match and money_context and monetary_tail(match):
        low, high = sorted((float(match.group(1).replace(",", "")), float(match.group(2).replace(",", ""))))
        return [(f"{low:g}-{high:g}", match.start(), match.end())]
    ceiling = re.search(
        r"\b(?:under|below|up to|at most|maximum(?: budget)?(?: of)?|max(?:imum)?|"
        r"no more than|not more than|less than|stretch to|cap(?:ped)? at)\s*" + _MONEY,
        clause,
    )
    floor = re.search(r"\b(?:at least|over|above|minimum(?: budget)?(?: of)?|more than)\s*" + _MONEY, clause)
    match = ceiling or floor
    if match and monetary_tail(match) and (money_context or not re.search(r"\b(?:size|length|inch|cm)\b", clause)):
        operator = "<=" if ceiling else ">="
        return [(f"{operator} {_number(match.group(1))}", match.start(), match.end())]
    if money_context:
        # Skip explicit size numbers before considering an unqualified budget.
        match = re.search(r"\$\s*(" + _NUMBER + r")|(" + _NUMBER + r")\s+dollars?\b", clause)
        if not match:
            match = re.search(r"\b(?:budget|spend|afford)\s*(?:(?:is|of|around|about)\s+)*" + _MONEY, clause)
        if not match and prompted:
            match = re.fullmatch(r"(?:around\s+|about\s+)?" + _MONEY + r"(?:\s+please)?", clause)
        if match:
            return [(f"<= {_number(next(value for value in match.groups() if value is not None))}", match.start(), match.end())]
    return []


def budget_hedged(clause: str, start: int, end: int) -> bool:
    """Whether a money span is stated approximately rather than as a firm limit.

    A hedged ceiling must not become a hard constraint. The guard would demote a
    product one cent over the stated figure below the entire candidate pool, and
    an approximate figure is not evidence that such a product is wrong.
    """
    window = clause[max(0, start - 20):end] + clause[end:end + 8]
    return bool(_BUDGET_HEDGE.search(window))


def _sizes(clause: str, prompted: bool) -> list[tuple[str, int, int]]:
    pattern = r"\b(?:size\s*(?:is|of|:)?\s*|(?:wear|wearing)\s+(?:a\s+)?)(?:(us|uk|eu)\s*)?(" + _SIZE_VALUE + r")(?!\w)"
    matches = list(re.finditer(pattern, clause))
    if not matches:
        matches = list(re.finditer(r"\b(us|uk|eu)\s*(?:size\s*)?(" + _SIZE_VALUE + r")(?!\w)", clause))
    if not matches and prompted:
        match = re.fullmatch(r"(?:a\s+)?(?:(us|uk|eu)\s*)?(" + _SIZE_VALUE + r")(?:\s+(?:please|would be (?:fine|good)))?[.!]?", clause)
        matches = [match] if match else []
    return [
        ((f"{match.group(1)} " if match.group(1) else "") + _SIZE_ALIASES.get(match.group(2), match.group(2)), match.start(), match.end())
        for match in matches
    ]


def _brands(clause: str) -> list[tuple[str, int, int]]:
    match = re.search(r"\b(?:brand(?: is|:)?|made by|from|by)\s+([a-z][a-z0-9 '&-]*)", clause)
    if not match:
        return []
    value = re.split(r"\b(?:in|with|for|that|which|and|or|please)\b", match.group(1))[0].strip()
    if any(value in aliases for aliases in _LOOKUPS.values()) or clause[:match.start()].endswith("made "):
        return []
    if 1 <= len(value.split()) <= 4 and value not in {"any", "anything", "the", "a", "me"}:
        return [(value, match.start(1), match.start(1) + len(value))]
    return []


def _residuals(clause: str, mentions: list[tuple[str, str, int, int]]) -> list[tuple[str, int, int]]:
    """Keep open-vocabulary descriptive spans, excluding already-owned facts.

    Function words separate independent phrases. Original offsets retain local
    negation scope, while masking known spans prevents corrected facts returning
    through a second, unstructured copy of their source sentence.
    """
    phrases: list[tuple[str, int, int]] = []
    discourse_spans = [(match.start(), match.end()) for match in re.finditer(
        r"\b(?:key|main|primary|essential)\s+(?:requirements?|preferences?)\b", clause)]
    tokens: list[re.Match[str]] = []

    def flush() -> None:
        if tokens:
            phrases.append((" ".join(token.group() for token in tokens), tokens[0].start(), tokens[-1].end()))
            tokens.clear()

    for token in re.finditer(r"\d+(?:\.\d+)?|[a-z][a-z0-9]*(?:['-][a-z0-9]+)*", clause):
        owned = any(token.start() < end and token.end() > start for _, _, start, end in mentions)
        owned |= any(token.start() < end and token.end() > start for start, end in discourse_spans)
        if owned or token.group() in _RESIDUAL_STOPWORDS or not re.search(r"[a-z]", token.group()):
            flush()
            continue
        if tokens and (_polarity(clause, token.start(), token.end()) != _polarity(clause, tokens[0].start(), tokens[0].end()) or len(tokens) >= 12):
            flush()
        tokens.append(token)
    flush()
    return phrases


def _quantities(clause: str, mentions: list[tuple[str, str, int, int]]) -> list[tuple[str, int, int]]:
    quantities = []
    for match in _QUANTITY_PATTERN.finditer(clause):
        if any(match.start() < end and match.end() > start for _, _, start, end in mentions):
            continue
        if match.start() and clause[match.start() - 1] == "$":
            continue
        number, unit = _number(match.group(1)), _UNITS.get(match.group(2), "")
        value = number + ("%" if unit == "%" else f" {unit}" if unit else "")
        quantities.append((value, match.start(), match.end()))
    return quantities


class SessionState:
    """Source-aware, session-local shopping preferences with explicit baselines.

    Profiles are retained as priors, not converted to current shopping intent.
    Ledger mode retracts contradicted facts, latest mode replaces each mentioned
    attribute, and history mode deliberately retains past positive facts.
    """

    def __init__(self, profile: dict, mode: str = "ledger", alternatives_mode: str = "off",
                 scoped_preferences: bool = False,
                 catalog_vocabulary: CatalogVocabulary | None = None,
                 canonical_state_semantics: bool = False) -> None:
        if mode not in {"ledger", "latest", "history"}:
            raise ValueError(f"Invalid state mode: {mode!r}")
        if alternatives_mode not in {"off", "parse", "grouped"}:
            raise ValueError(f"Invalid alternatives mode: {alternatives_mode!r}")
        if alternatives_mode == "grouped" and mode != "ledger":
            raise ValueError("Grouped alternatives require ledger state")
        self.profile = deepcopy(profile)
        self.mode = mode
        self.alternatives_mode = alternatives_mode
        self.scoped_preferences = scoped_preferences
        self.catalog_vocabulary = catalog_vocabulary
        self.canonical_state_semantics = canonical_state_semantics
        self.unsupported_alternatives: list[dict[str, str]] = []
        self.history: list[SourceRecord] = []
        self.preferences: list[Preference] = []
        self.last_question: str | None = None
        self.last_question_goal: str | None = None
        self.asked_counts: dict[str, int] = {}
        self.asked_question_goals: set[str] = set()
        self.unproductive_attributes: set[str] = set()
        self.last_update_informative = False
        self.last_answer_productivity = "not_applicable"
        self.last_feedback = FeedbackDecision("none")
        self.last_override = OverrideDecision()
        self.revision = 0
        self.turn = 0

    def record_question(self, attribute: str | None, goal: str | None = None) -> None:
        self.last_question = attribute
        self.last_question_goal = goal
        if attribute is not None:
            self.asked_counts[attribute] = self.asked_counts.get(attribute, 0) + 1
        if goal is not None:
            self.asked_question_goals.add(goal)

    def active_preferences(self) -> list[Preference]:
        return [preference for preference in self.preferences if preference.active]

    def semantic_signature(self) -> tuple[tuple, ...]:
        """Canonical active facts without turn, source text, or group-ID churn."""
        active = self.active_preferences()
        group_values: dict[str, tuple[str, ...]] = {}
        for preference in active:
            if preference.alternative_group is not None:
                group_values[preference.alternative_group] = tuple(sorted({
                    item.value for item in active
                    if item.attribute == preference.attribute
                    and item.alternative_group == preference.alternative_group
                    and item.polarity == 1
                }))
        return tuple(sorted(
            (
                preference.attribute,
                preference.value,
                preference.polarity,
                preference.hard,
                round(preference.confidence, 6),
                preference.depends_on,
                group_values.get(preference.alternative_group),
                preference.scope,
            )
            for preference in active
        ))

    def effective_preferences(self, decay_turns: int = 0) -> list[Preference]:
        """Return ranking evidence with decay applied only to inferred soft signals."""
        result = deepcopy(self.active_preferences())
        if decay_turns <= 0:
            return result
        for preference in result:
            if preference.source_kind == "inferred" and not preference.hard and preference.polarity == 1:
                age = max(0, self.turn - preference.source_turn)
                preference.confidence *= 0.5 ** (age / decay_turns)
        return result

    def _signature(self) -> set[tuple[str, str, int, bool, float, tuple[str, str] | None,
                                    str | None, str | None, str]]:
        return {(p.attribute, p.value, p.polarity, p.hard, p.confidence, p.depends_on,
                 p.alternative_group, p.scope, p.source_kind)
                for p in self.active_preferences()}

    def _override_facts(self) -> set[OverrideFact]:
        """Project active state onto semantic facts, excluding source churn."""
        return {
            OverrideFact(p.attribute, p.value, p.polarity, p.scope)
            for p in self.active_preferences() if p.polarity != 0
        }

    @staticmethod
    def _attribute_change(match: re.Match[str] | None) -> str | None:
        if match is None:
            return None
        value = next((group for group in match.groups() if group), None)
        return {
            "fabric": "material", "colour": "color", "product type": "category",
            "features": "feature",
        }.get(value, value)

    def _decide_override(self, normalized: str, before: set[OverrideFact],
                         assertions: list[_Assertion], requested_attribute: str | None) -> OverrideDecision:
        after = self._override_facts()
        retired = before - after
        added = after - before
        retained = before & after
        retired_attributes = {fact.attribute for fact in retired}
        added_attributes = {fact.attribute for fact in added}
        changed = retired_attributes | ({requested_attribute} if requested_attribute else set())
        reasons: list[str] = []
        semantic = False
        if retired:
            semantic = True
            reasons.append("preference_retracted")
        if retired_attributes & added_attributes:
            reasons.append("attribute_replaced")
        if any(old.attribute == new.attribute and old.value == new.value and old.polarity != new.polarity
               for old in retired for new in added):
            reasons.append("polarity_changed")
        if requested_attribute:
            semantic = True
            reasons.append("attribute_change_requested")

        protected = bool(_NO_CHANGE_GUARD.search(normalized) or _EITHER_OR_GUARD.search(normalized))
        explicit_directive = bool(_EXPLICIT_OVERRIDE.search(normalized) and not protected)
        marker = bool(explicit_directive or _POSSIBLE_OVERRIDE.search(normalized))
        meaningful_assertion = any(item.preference.polarity != 0 for item in assertions)
        phrase_restatement = bool(before and marker and meaningful_assertion and not protected and not semantic)
        if explicit_directive:
            reasons.append("explicit_override_directive")
        if phrase_restatement:
            reasons.append("explicit_correction_restatement")
        detected = semantic or explicit_directive or phrase_restatement
        confidence = 0.98 if semantic and (retired_attributes & added_attributes) else (
            0.92 if semantic else 0.85 if explicit_directive else 0.75 if phrase_restatement else 0.0
        )
        return OverrideDecision(
            detected=detected,
            confidence=confidence,
            changed_attributes=tuple(sorted(changed)),
            retired=tuple(sorted(retired, key=lambda fact: (fact.attribute, fact.value, fact.polarity, fact.scope or ""))),
            added=tuple(sorted(added, key=lambda fact: (fact.attribute, fact.value, fact.polarity, fact.scope or ""))),
            retained=tuple(sorted(retained, key=lambda fact: (fact.attribute, fact.value, fact.polarity, fact.scope or ""))),
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def update(self, user_message: str, turn: int) -> None:
        self.turn = turn
        self.unsupported_alternatives = []
        before = self._signature()
        override_before = self._override_facts()
        normalized = _normalize(user_message)
        no_change_guard = bool(_NO_CHANGE_GUARD.search(normalized))
        attribute_change_match = None if no_change_guard else _ATTRIBUTE_CHANGE_REQUEST.search(normalized)
        requested_attribute = self._attribute_change(attribute_change_match)
        attribute_rejection = _ATTRIBUTE_REJECTION.search(normalized)
        parse_message = user_message
        if _PRODUCT_TYPE_REJECTION.search(normalized):
            self.last_feedback = FeedbackDecision("product_type", "category", "generic_product_type_rejection")
            for preference in self.active_preferences():
                if preference.attribute == "category" and preference.polarity == 1:
                    preference.active = False
            parse_message = _PRODUCT_TYPE_REJECTION.sub("", normalized)
        elif _ITEM_REJECTION.search(normalized):
            self.last_feedback = FeedbackDecision("item", None, "item_only_rejection")
            parse_message = _ITEM_REJECTION.sub("", normalized)
        elif attribute_rejection:
            attribute = {"fabric": "material", "colour": "color"}.get(
                attribute_rejection.group(1), attribute_rejection.group(1),
            )
            self.last_feedback = FeedbackDecision("attribute_unknown", attribute,
                                                  "attribute_rejection_without_value")
            parse_message = _ATTRIBUTE_REJECTION.sub("", normalized)
        elif requested_attribute:
            self.last_feedback = FeedbackDecision(
                "attribute_unknown", requested_attribute, "attribute_change_without_value",
            )
            for preference in self.active_preferences():
                if preference.attribute == requested_attribute and preference.polarity == 1:
                    preference.active = False
        else:
            self.last_feedback = FeedbackDecision("none")
        assertions = self._extract(parse_message, turn)
        negative = next((item.preference for item in assertions if item.preference.polarity == -1), None)
        if negative is not None and self.last_feedback.scope not in {"item", "product_type"}:
            self.last_feedback = FeedbackDecision("attribute_value", negative.attribute,
                                                  "explicit_negative_value")
        grouped: dict[int, dict[str, list[_Assertion]]] = defaultdict(lambda: defaultdict(list))
        for assertion in assertions:
            # The control modes retain their whole-message attribute groups.
            clause = assertion.clause if self.mode == "ledger" else 0
            grouped[clause][assertion.preference.attribute].append(assertion)
        for clause in sorted(grouped):
            for attribute, group in grouped[clause].items():
                self._apply(attribute, group)
                if any(item.preference.polarity != 0 for item in group):
                    self.unproductive_attributes.discard(attribute)
                elif self.mode == "ledger" and any(item.preference.value == "any" for item in group):
                    self.unproductive_attributes.add(attribute)
            # Retractions take effect before a later clause can restore an owner;
            # restoring the material or feature must not revive its old quantity.
            owners = {(p.attribute, p.value) for p in self.active_preferences() if p.polarity == 1 and p.depends_on is None}
            for preference in self.active_preferences():
                if preference.polarity == 1 and preference.depends_on is not None and preference.depends_on not in owners:
                    preference.active = False
        self.last_update_informative = self._signature() != before
        self.last_override = self._decide_override(
            normalized, override_before, assertions, requested_attribute,
        )
        if self.last_question is None:
            self.last_answer_productivity = "not_applicable"
        elif any(item.preference.polarity == 0 for item in assertions):
            self.last_answer_productivity = "neutral"
        elif self.last_update_informative:
            self.last_answer_productivity = (
                "contradictory" if self.last_feedback.scope in {"product_type", "attribute_value"}
                or self.last_override.detected else "productive"
            )
        elif self.last_question in self.unproductive_attributes:
            self.last_answer_productivity = "neutral"
        else:
            self.last_answer_productivity = "unresolved"
        if self.last_update_informative:
            self.revision += 1
        self.history.append(SourceRecord(turn, user_message, [a.preference for a in assertions], self.last_update_informative))

    def _extract(self, message: str, turn: int) -> list[_Assertion]:
        text = re.sub(r"\b(?:anything|everything) but\b", "without", _normalize(message))
        if self.alternatives_mode != "off":
            text = re.sub(r"\bneither\b", "not", text)
        turn_replacement = bool(
            self.canonical_state_semantics
            and (_REPLACEMENT.search(text) or _IMPLICIT_PREFERENCE_SHIFT.search(text))
        )
        result: list[_Assertion] = []
        residual_assertions: list[_Assertion] = []
        qualified_component_contrast = bool(_NOT_JUST_COMPONENT.search(text))
        clauses = re.split(
            r"(?<!\d)\.(?!\d)|[;!?]|,(?!\d)|\bbut\b|"
            r"\band\s+(?=(?:i\s+(?:need|want|prefer|don't|do not)|it\s+(?:must|needs|should)))",
            text,
        )
        unproductive = False
        pending_replacement = False
        for clause_index, clause in enumerate(clauses):
            clause = clause.strip()
            if not clause:
                continue
            attributes = _mentioned_attributes(clause)
            if _NO_NEW_INFORMATION.search(clause):
                self.unproductive_attributes.update(attributes | ({self.last_question} if self.last_question else set()))
                unproductive = True
                continue
            mentions = [(attribute, _LOOKUPS[attribute][match.group()], match.start(), match.end())
                        for attribute, pattern in _PATTERNS.items() for match in pattern.finditer(clause)]
            catalog_sources: dict[tuple[str, str, int, int], VocabularyMatch] = {}
            if self.catalog_vocabulary is not None:
                occupied = [(start, end) for _, _, start, end in mentions]
                for match in self.catalog_vocabulary.find(clause, occupied):
                    key = (match.attribute, match.canonical, match.start, match.end)
                    mentions.append(key)
                    catalog_sources[key] = match
            alternatives = self._alternatives(clause, mentions) if self.alternatives_mode != "off" else {}
            choice_replacements = self._choice_replacements(clause, mentions) if self.alternatives_mode == "grouped" else set()
            no_preference = (
                _NO_PREFERENCE.search(clause)
                or re.search(r"\bdon't have (?:a |any )?preference\b", clause)
                or (_CANONICAL_NO_PREFERENCE.search(clause)
                    if self.canonical_state_semantics else None)
            )
            if self.alternatives_mode != "off" and no_preference and re.match(r"(?:any|either)\b", no_preference.group()):
                # Stop at this acceptance segment, not a later independent list.
                no_preference = re.search(r"\b(?:any|either)\b.*?\b(?:fine|okay|ok|works|will do)\b", clause)
            listed_acceptance = no_preference and re.match(r"(?:any|either)\b", no_preference.group()) and any(
                value in alternatives.get(attribute, set()) and no_preference.start() <= start < no_preference.end()
                for attribute, value, start, _ in mentions
            )
            if no_preference and not listed_acceptance:
                neutral_attributes = attributes or ({self.last_question} if self.last_question else set())
                scoped_neutral = _mentioned_attributes(no_preference.group()) if self.alternatives_mode != "off" else set()
                independent_alternatives = bool(scoped_neutral and alternatives and scoped_neutral.isdisjoint(alternatives))
                if independent_alternatives:
                    neutral_attributes = scoped_neutral
                    mentions = [mention for mention in mentions
                                if not no_preference.start() <= mention[2] < no_preference.end()]
                for attribute in sorted(neutral_attributes):
                    if attribute == "other":
                        # A catch-all answer declines extra detail; it does not
                        # retract the user's existing open-vocabulary requests.
                        unproductive = True
                        continue
                    result.append(_Assertion(Preference(attribute, "any", turn, message, polarity=0), clause=clause_index))
                self.unproductive_attributes.update(neutral_attributes)
                if not independent_alternatives:
                    continue
            unproductive_match = _UNPRODUCTIVE.search(clause)
            if unproductive_match and not _ACTIONABLE_REJECTION.search(clause):
                self.unproductive_attributes.update(attributes or ({self.last_question} if self.last_question else set()))
                unproductive = True
                continue

            mentions.extend(("budget", value, start, end) for value, start, end in _budget(clause, self.last_question == "budget"))
            mentions.extend(("size", value, start, end) for value, start, end in _sizes(clause, self.last_question == "size"))
            mentions.extend(("brand", value, start, end) for value, start, end in _brands(clause))
            quantities = _quantities(clause, mentions)

            additive = bool(_ADDITIVE.search(clause))
            replacement = bool(
                _REPLACEMENT.search(clause)
                or _IMPLICIT_PREFERENCE_SHIFT.search(clause)
                or pending_replacement
                or (turn_replacement and not _ADDITIVE.search(clause))
            )
            soft = bool(_SOFT.search(clause))
            discourse = [("discourse", "", match.start(), match.end()) for match in re.finditer(r"\bworks\b", clause)] if alternatives else []
            residuals = _residuals(clause, mentions + discourse + [("quantity", value, start, end) for value, start, end in quantities])
            pending_replacement = replacement and not mentions and not residuals
            owner_spans = [(attribute, value, start, end) for attribute, value, start, end in mentions if attribute not in {"budget", "size"}]
            for attribute, value, start, end in mentions:
                catalog_match = catalog_sources.get((attribute, value, start, end))
                polarity = _polarity(clause, start, end)
                if _SUFFIX_RELAXATION.search(clause[end:]):
                    polarity = 0
                hard = polarity == -1 or bool(_HARD.search(clause)) or (
                    attribute == "budget" and not soft and not budget_hedged(clause, start, end))
                if catalog_match is not None:
                    hard = polarity == -1
                if polarity == 0:
                    hard = False
                confidence = 0.8 if soft and not hard else 1.0
                if catalog_match is not None:
                    confidence = min(confidence, catalog_match.confidence)
                replace_material = attribute == "material" and not additive and bool(re.search(r"\bprefer\b", clause))
                alternative_group = (
                    (
                        "choice:" + attribute + ":" + hashlib.sha256(
                            "\0".join(sorted(alternatives[attribute])).encode("utf-8")
                        ).hexdigest()[:16]
                        if self.canonical_state_semantics
                        else f"{turn}:{clause_index}:{attribute}"
                    )
                    if self.alternatives_mode == "grouped" and polarity == 1 and value in alternatives.get(attribute, set())
                    else None
                )
                component = (_COMPONENT_AFTER.match(clause[end:])
                             if self.scoped_preferences and attribute in {"material", "color", "feature"} else None)
                scope = component.group(1) if component else None
                result.append(_Assertion(Preference(attribute, value, turn, message, hard=hard,
                                                   polarity=polarity, confidence=confidence,
                                                   alternative_group=alternative_group, scope=scope,
                                                   source_kind=catalog_match.provenance if catalog_match else "explicit"),
                                         additive or catalog_match is not None,
                                         replacement or replace_material, clause=clause_index,
                                         choice_replacement=attribute in choice_replacements))
            for value, start, end in residuals:
                if self.canonical_state_semantics and value in {"correction", "suitable"}:
                    continue
                not_just_match = _NOT_JUST_COMPONENT.search(clause)
                if not_just_match and not_just_match.start() < start:
                    # The component is insufficient on its own, not forbidden.
                    # Keep it out of the preference ledger and retain the
                    # qualified relation that appeared before the contrast.
                    continue
                polarity = 0 if _SUFFIX_RELAXATION.search(clause[end:]) else _polarity(clause, start, end)
                qualified_materials = [material for attribute, material, _, material_end in mentions
                                       if qualified_component_contrast and material_end <= start
                                       and attribute == "material" and not clause[material_end:start].strip(" -")]
                existing_attributes = {p.attribute for p in self.active_preferences() if p.value == value}
                residual_attribute = (
                    next(iter(attributes)) if len(attributes) == 1 else
                    next(iter(existing_attributes)) if len(existing_attributes) == 1 else "other"
                )
                residual_assertions.append(_Assertion(
                    Preference(residual_attribute, value, turn, message, hard=polarity == -1,
                               polarity=polarity, confidence=0.65, source_kind="inferred"),
                    additive, replacement, clause=clause_index,
                ))
                if qualified_materials:
                    residual_assertions.append(_Assertion(
                        Preference("other", f"{qualified_materials[-1]} {value}", turn, message,
                                   hard=polarity == -1, polarity=polarity, confidence=0.8),
                        additive, replacement, clause=clause_index,
                    ))
                owner_spans.append((residual_attribute, value, start, end))
            for quantity, start, end in quantities:
                following = [owner for owner in owner_spans if owner[2] >= end and not clause[end:owner[2]].strip(" -")]
                preceding = [owner for owner in owner_spans if owner[3] <= start and not clause[owner[3]:start].strip(" -")]
                owner = min(following, key=lambda item: item[2]) if following else max(preceding, key=lambda item: item[3]) if preceding else None
                if owner is not None:
                    polarity = _polarity(clause, start, owner[3] if following else end)
                    result.append(_Assertion(Preference(
                        "other", f"{quantity} {owner[1]}", turn, message, hard=polarity == -1,
                        polarity=polarity, confidence=0.8, depends_on=(owner[0], owner[1]),
                    ), clause=clause_index))
        if not result and not unproductive:
            fallback = self._prompted_value(text)
            if fallback is not None:
                result.append(_Assertion(Preference(self.last_question, fallback, turn, message, confidence=0.8)))
                return result
        result.extend(residual_assertions)
        if not result and self.last_question is not None:
            self.unproductive_attributes.add(self.last_question)
        return result

    def _unsupported(self, attribute: str, reason: str) -> None:
        diagnostic = {"attribute": attribute, "reason": reason}
        if diagnostic not in self.unsupported_alternatives and len(self.unsupported_alternatives) < 8:
            self.unsupported_alternatives.append(diagnostic)

    def _choice_replacements(self, clause: str, mentions: list[tuple[str, str, int, int]]) -> set[str]:
        """Extract selection syntax; the live ledger decides whether it replaces a group."""
        if re.search(r"\bor\b", clause):
            return set()
        replaced = set()
        for attribute in {mention[0] for mention in mentions}:
            values = sorted((m for m in mentions if m[0] == attribute and _polarity(clause, m[2], m[3]) == 1),
                            key=lambda item: item[2])
            if any(re.match(r"\s+only\b", clause[end:]) or re.search(r"(?<!not )\bonly\s+$", clause[:start])
                   for _, _, start, end in values):
                replaced.add(attribute)
            if len(values) > 1 and all(re.fullmatch(r"\s+and\s+", clause[first[3]:second[2]])
                                       for first, second in zip(values, values[1:])):
                replaced.add(attribute)
        return replaced

    def _alternatives(self, clause: str, mentions: list[tuple[str, str, int, int]]) -> dict[str, set[str]]:
        """Recognize direct, positive known-value OR chains, not a Boolean grammar."""
        operators = list(re.finditer(r"\bor\b", clause))
        if not operators:
            return {}
        positive = [mention for mention in mentions if _polarity(clause, mention[2], mention[3]) == 1]
        if not positive:
            return {}
        attributes = sorted({mention[0] for mention in positive})
        reason = "nested alternatives are unsupported" if re.search(r"[()\[\]{}]", clause) else None
        linked: dict[str, set[str]] = defaultdict(set)
        covered: set[int] = set()
        for attribute in attributes:
            values = sorted((mention for mention in positive if mention[0] == attribute), key=lambda item: item[2])
            link_count = 0
            for first, second in zip(values, values[1:]):
                gap = clause[first[3]:second[2]]
                if first[1] != second[1] and re.fullmatch(r"\s+or\s+(?:(?:a|an|the)\s+)?", gap):
                    linked[attribute].update((first[1], second[1]))
                    covered.add(first[3] + gap.index("or"))
                    link_count += 1
            if attribute in linked and link_count != len(values) - 1:
                reason = reason or "mixed alternatives are unsupported"
        if len(linked) != 1 or covered != {operator.start() for operator in operators}:
            reason = reason or "cross-attribute or nonadjacent alternatives are unsupported"
        if reason:
            for attribute in attributes:
                self._unsupported(attribute, reason)
            return {}
        return dict(linked)

    def _prompted_value(self, clause: str) -> str | None:
        if self.last_question not in {"brand", "material", "color", "style", "feature", "use_case", "category", "other"}:
            return None
        if re.search(r"\b(?:no|not|don't|none|nothing|anything|whatever|yes|okay|show|options|recommendations|pick|choose|something|actually)\b", clause):
            return None
        descriptive = _residuals(clause, [])
        if not descriptive:
            return None
        value = clause[descriptive[0][1]:descriptive[-1][2]]
        if 1 <= len(value.split()) <= 6 and re.fullmatch(r"[a-z0-9][a-z0-9 '&/-]*", value):
            return value
        return None

    def _apply(self, attribute: str, assertions: list[_Assertion]) -> None:
        incoming = [assertion.preference for assertion in assertions]
        if self.mode != "history":
            neutral_positions = [index for index, p in enumerate(incoming) if p.polarity == 0 and p.value == "any"]
            if neutral_positions:
                last_neutral = neutral_positions[-1]
                incoming = incoming[last_neutral + 1:] or [incoming[last_neutral]]
        previous = [p for p in self.active_preferences() if p.attribute == attribute]
        if self.alternatives_mode == "grouped" and self.mode == "ledger":
            if any(p.polarity == 1 and p.alternative_group is not None for p in previous):
                for assertion in assertions:
                    assertion.replacement |= assertion.choice_replacement
            if not self._prepare_alternative_update(attribute, incoming, previous, assertions):
                return
        if self.mode == "latest":
            for preference in previous:
                preference.active = False
        elif self.mode == "ledger":
            positives = {p.value for p in incoming if p.polarity == 1}
            negatives = {p.value for p in incoming if p.polarity == -1}
            neutral = any(p.polarity == 0 and p.value == "any" for p in incoming)
            relaxed = {p.value for p in incoming if p.polarity == 0 and p.value != "any"}
            replacement = any(assertion.replacement for assertion in assertions)
            additive = any(assertion.additive for assertion in assertions) and not replacement
            replace_slot = bool(positives) and not additive and attribute not in {"feature", "use_case", "other"}
            if attribute == "material" and (not replacement or negatives):
                replace_slot = False
            changed_owners = {p.depends_on for p in incoming if p.depends_on is not None}
            retained_groups = {p.alternative_group for p in incoming if p.alternative_group is not None}
            for preference in previous:
                excluded = any(
                    candidate.value == preference.value
                    and (candidate.scope is None or preference.scope is None or candidate.scope == preference.scope)
                    for candidate in incoming if candidate.polarity == -1
                ) or (attribute == "other" and any(
                    re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", preference.value) for value in negatives
                ))
                superseded_quantity = preference.depends_on in changed_owners and preference.value not in positives | negatives
                if neutral or superseded_quantity or preference.value in relaxed or (preference.polarity == 0 and (preference.value == "any" or preference.value in positives | negatives)):
                    preference.active = False
                elif preference.polarity == 1 and (excluded or (replace_slot and preference.value not in positives
                                                               and preference.alternative_group not in retained_groups)):
                    preference.active = False
                elif preference.polarity == -1 and any(
                    candidate.value == preference.value
                    and (candidate.scope is None or preference.scope is None or candidate.scope == preference.scope)
                    for candidate in incoming if candidate.polarity == 1
                ):
                    preference.active = False
        for preference in incoming:
            duplicate = next((p for p in self.active_preferences() if p.attribute == attribute
                              and p.value == preference.value and p.polarity == preference.polarity
                              and p.depends_on == preference.depends_on and p.scope == preference.scope), None)
            if duplicate is None:
                self.preferences.append(preference)
            else:
                preference.hard = preference.hard or duplicate.hard
                preference.confidence = max(preference.confidence, duplicate.confidence)
                duplicate.active = False
                self.preferences.append(preference)

    def _prepare_alternative_update(self, attribute: str, incoming: list[Preference], previous: list[Preference],
                                    assertions: list[_Assertion]) -> bool:
        groups: dict[str, list[Preference]] = defaultdict(list)
        for preference in previous:
            if preference.polarity == 1 and preference.alternative_group is not None:
                groups[preference.alternative_group].append(preference)
        replacements = any(assertion.replacement for assertion in assertions)
        choices = [p for p in incoming if p.polarity == 1 and p.alternative_group is not None]
        values = {p.value for p in choices}
        if choices and not replacements:
            for group_id, members in groups.items():
                old_values = {p.value for p in members}
                if values & old_values and values != old_values:
                    self._unsupported(attribute, "overlapping alternatives require an explicit replacement")
                    for preference in choices:
                        preference.active = False
                    incoming[:] = [p for p in incoming if p not in choices]
                    return bool(incoming)
                if values == old_values:
                    for preference in choices:
                        preference.alternative_group = group_id
        positives = [p for p in incoming if p.polarity == 1]
        for preference in positives:
            duplicate = next((p for p in previous if p.polarity == 1 and p.value == preference.value
                              and p.alternative_group is not None), None)
            if duplicate is not None:
                preference.hard |= any(p.hard for p in groups[duplicate.alternative_group])
                preference.confidence = max(preference.confidence, duplicate.confidence)
                if not replacements and preference.alternative_group is None:
                    preference.alternative_group = duplicate.alternative_group
        # A requirement belongs to the choice set, even when its hard wording
        # was repeated on one member. Retraction must not lose that force.
        for group_id, members in groups.items():
            restatements = [p for p in positives if p.alternative_group == group_id]
            if any(p.hard for p in members + restatements):
                for preference in members + restatements:
                    preference.hard = True
        if replacements and positives:
            additive = any(assertion.additive for assertion in assertions)
            positive_values = {p.value for p in positives}
            for members in groups.values():
                if choices or not additive or any(p.value in positive_values for p in members):
                    for preference in members:
                        preference.active = False
        return True

    def query(self) -> str:
        # Never search rejected values, neutral answers, or raw conversational
        # scaffolding. Rebuilding from active facts also retracts old source text.
        active = self.active_preferences()
        if self.canonical_state_semantics:
            active = sorted(
                active,
                key=lambda preference: (
                    preference.attribute, preference.value, preference.polarity,
                    preference.scope or "", preference.depends_on or ("", ""),
                ),
            )
        expanded = {p.depends_on for p in active if p.polarity == 1 and p.depends_on is not None}
        values = dict.fromkeys(p.value for p in active if p.polarity == 1 and p.attribute != "budget"
                               and (p.attribute, p.value) not in expanded)
        return " ".join(values)

    def source_alias_query(self) -> str:
        """Return exact non-canonical parser aliases from current positive facts."""
        aliases: dict[str, None] = {}
        for preference in self.active_preferences():
            if preference.polarity != 1 or preference.depends_on is not None:
                continue
            for alias in _VALUES.get(preference.attribute, {}).get(preference.value, ()):
                if alias == preference.value:
                    continue
                if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", _normalize(preference.source_text)):
                    aliases.setdefault(alias, None)
        return " ".join(aliases)
