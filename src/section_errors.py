"""Prose for "Data Analysis and Results": where the models fail, and the two checks.

Covers the large-error analysis (Objective 5), what the model leans on (Objective 6),
and the two robustness checks. The description of the badly predicted cars is built
from the measured gaps, ranked largest first, not from any expectation about them.
"""
from __future__ import annotations


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _large_error_blocks(F: dict) -> list[tuple]:
    ranked = F["profile_ranked"]
    top = ranked[0]
    newer = F["big_age"] < F["rest_age"]

    bullets = [
        f"{F['big_n']:,} held-back cars ({F['big_pct']:.1f}%) were missed by more "
        f"than ${F['large_threshold']:,.0f}, the line fixed in advance from training "
        f"data.",
        f"The single biggest difference is {top['label'].lower()}: "
        f"{_pct(top['large'])} of the badly predicted cars against "
        f"{_pct(top['rest'])} of the rest, a gap of {top['diff']*100:+.1f} percentage "
        f"points.",
        f"Gaps in the advert follow close behind - "
        + "; ".join(f"{r['label'].lower()} {r['diff']*100:+.1f} points"
                    for r in ranked[1:4])
        + " - so missing information matters, but being an unusual car matters more.",
        f"They are {'younger' if newer else 'older'} than the rest "
        f"({F['big_age']:.1f} years against {F['rest_age']:.1f}) and have "
        f"{'lower' if F['big_odo'] < F['rest_odo'] else 'higher'} mileage "
        f"({F['big_odo']:,.0f} against {F['rest_odo']:,.0f}), which is the opposite of "
        f"the worn-out old car one might expect to be hard to price.",
        f"They ask far more on average (${F['big_price']:,.0f} against "
        f"${F['rest_price']:,.0f}). Part of that follows automatically from using a "
        f"dollar threshold, since a large dollar miss is easier to incur on an "
        f"expensive car - which is exactly why the percentage measure was also needed.",
    ]
    salvage = next((r for r in ranked if "salvage" in r["label"].lower()), None)
    if salvage is not None and salvage["diff"] < 0:
        bullets.append(
            f"Written-off and rebuilt cars are actually rarer among the badly "
            f"predicted ({_pct(salvage['large'])} against {_pct(salvage['rest'])}), "
            f"for the same reason: such cars are cheap, and a dollar threshold "
            f"seldom flags cheap cars.")
    bullets.append(
        "A large miss is not evidence of a faulty advert. These are mostly newer, "
        "dearer, less common vehicles with fewer comparable cars in the data, and "
        "treating every large error as bad data would be a mistake.")

    topic = ("The worst-predicted cars are a recognisable type - newer, dearer, "
             "lower-mileage and uncommon - rather than simply cars with gaps in their "
             "adverts.")
    return [("topic", topic), ("bullets", bullets)]


def _extreme_blocks(F: dict) -> list[tuple]:
    """The very worst misses, separated into model limits and data limits."""
    ratio = (F["junk_share_big"] / F["junk_share_all"]) if F["junk_share_all"] else 0
    return [
        ("topic", "The very largest misses are of two distinct kinds, and only one of "
                  "them is the model's fault."),
        ("bullets", [
            "Most of them are genuine exotics - a Mercedes G-Class, two Porsche 911s, "
            "a Ferrari - advertised far above anything comparable in the data. With "
            "few such cars to learn from, the model prices them towards the ordinary.",
            "The rest are adverts whose price was typed rather than meant. Three of "
            "the eight biggest misses ask exactly $123,456. For one of them, a "
            "24-year-old Chevrolet Tahoe with 270,000 miles, the model predicted "
            "$3,538 - which is roughly what such a car is worth. The model was right "
            "and the advert was wrong.",
            f"Such prices are rare overall: {F['junk_price_cars']} of the held-back "
            f"cars ({F['junk_share_all']*100:.3f}%). But they are about "
            f"{ratio:.0f} times over-represented among the large errors "
            f"({F['junk_share_big']*100:.2f}%), so they matter at the extreme while "
            f"being negligible in the bulk.",
            f"Expensive cars dominate the large errors instead: "
            f"{F['dear_share_big']*100:.1f}% of them ask more than "
            f"${F['dear_threshold']:,} against {F['dear_share_all']*100:.1f}% of all "
            f"held-back cars - roughly a twelvefold concentration.",
            f"The model priced {F['underpriced_share_big']*100:.1f}% of large-error "
            f"cars below their advert, which is the same hedging seen in the bands, "
            f"appearing again at the extreme.",
            f"The rule used to flag a typed price is deliberately narrow: digits "
            f"running in sequence only. Repeated digits such as $99,999 were "
            f"considered and rejected, because that is ordinary pricing exactly like "
            f"the $69,999 beside it. The rule therefore under-counts, and other "
            f"nonsense prices certainly survive - which is itself a limitation of "
            f"using advertised prices. The values found were "
            f"{', '.join(f'${p:,}' for p in F['junk_prices_found'])}.",
        ]),
    ]


def _importance_blocks(F: dict) -> list[tuple]:
    imp, miss, L = F["importance"], F["feature_missing"], F["label"]
    order = sorted(imp, key=imp.get, reverse=True)
    first, second, third, fourth = order[0], order[1], order[2], order[3]

    bullets = [
        "Reliance is measured by scrambling one column at a time and recording how "
        "much the average miss worsens (permutation importance).",
        f"Scrambling {L(first)} adds ${imp[first]:,.0f} to the average miss, and "
        f"{L(second)} adds ${imp[second]:,.0f}. Together they dominate every other "
        f"characteristic.",
        f"Both are almost always recorded - {L(first)} is blank in {miss[first]:.1f}% "
        f"of adverts and {L(second)} in {miss[second]:.1f}% - so the model's two main "
        f"supports are reliably available.",
        f"The next two are a different story. {L(third).capitalize()} "
        f"(${imp[third]:,.0f}) and {L(fourth)} (${imp[fourth]:,.0f}) are blank in "
        f"{miss[third]:.1f}% and {miss[fourth]:.1f}% of adverts, so for a large share "
        f"of cars the model is deprived of information it would otherwise lean on "
        f"heavily - which links directly to the errors above.",
        f"Condition is the field this literature discusses most, yet scrambling it "
        f"costs only ${imp.get('condition', 0):,.0f}, far less than {L(first)} or "
        f"{L(second)}. Sellers report it inconsistently, which limits what the model "
        f"can take from it.",
        "This describes what the model leans on, not what causes prices. It does not "
        "show that changing a car's mileage would change its asking price.",
    ]
    return [("topic", f"The model leans most heavily on {L(first)} and {L(second)}, "
                      f"but several of the characteristics it relies on next are the "
                      f"ones most often left blank."),
            ("bullets", bullets)]


def blocks(F: dict) -> list[tuple]:
    out = _large_error_blocks(F)
    out += [
        ("table", "profile",
         "Table 5. Badly predicted cars compared with all the rest."),
        ("figure", "fig14_large_error_profile.png",
         "Figure 12. Characteristics that separate badly predicted cars from the "
         "rest."),
    ]
    out += _extreme_blocks(F)
    out += [("table", "extremes",
             "Table 6. The eight largest misses, and whether the advertised price "
             "looks typed rather than meant.")]
    out += _importance_blocks(F)
    out += [
        ("table", "importance",
         "Table 7. How much accuracy each characteristic accounts for, against how "
         "often it is missing."),
        ("figure", "fig13_permutation_importance.png",
         "Figure 13. Dollars added to the average miss when each column is "
         "scrambled."),
    ]
    return out
