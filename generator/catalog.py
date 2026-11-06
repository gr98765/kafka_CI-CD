"""The 'world' the generator draws from: spending categories, countries, merchants.

Kept separate from generate.py so the numbers you may want to tweak
(typical prices, how common each category is) live in one small file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    median_usd: float  # typical payment size in US dollars
    weight: float  # how common this kind of spending is
    suffixes: tuple[str, ...]  # used to build merchant names


CATEGORIES: dict[str, Category] = {
    "groceries": Category(45, 0.20, ("Grocers", "Market", "Foods")),
    "dining": Category(22, 0.22, ("Cafe", "Kitchen", "Grill")),
    "fuel": Category(40, 0.10, ("Fuel", "Gas & Go")),
    "retail": Category(55, 0.15, ("Goods", "Outfitters", "Store")),
    "travel": Category(220, 0.03, ("Airlines", "Hotels", "Rail")),
    "subscriptions": Category(12, 0.10, ("Stream", "Cloud", "Premium")),
    "utilities": Category(90, 0.05, ("Power", "Water", "Telecom")),
    "entertainment": Category(25, 0.09, ("Cinemas", "Games", "Live")),
    "health": Category(60, 0.06, ("Pharmacy", "Clinic", "Dental")),
}

# country code -> (share of accounts, currency, US dollars per 1 unit)
COUNTRIES: dict[str, tuple[float, str, float]] = {
    "US": (0.70, "USD", 1.00),
    "GB": (0.10, "USD", 1.00),
    "DE": (0.08, "USD", 1.00),
    "CA": (0.07, "USD", 1.00),
    "MX": (0.05, "USD", 1.00),
}  # everything is in US dollars for now; countries only matter for location rules

PREFIXES: tuple[str, ...] = (
    "Maple", "Harbor", "Summit", "Cedar", "Blue", "Golden", "River", "Urban",
    "Pine", "Northern", "Sunrise", "Willow", "Iron", "Bright", "Oak", "Coastal",
    "Silver", "Prairie", "Lakeside", "Redwood",
)  # fmt: skip


@dataclass(frozen=True)
class Merchant:
    merchant_id: str
    name: str
    category: str
    country: str


def build_merchants(rng: random.Random, per_group: int = 4) -> list[Merchant]:
    """Make `per_group` merchants for every (country, category) pair."""
    merchants: list[Merchant] = []
    for country in COUNTRIES:
        for category, info in CATEGORIES.items():
            for _ in range(per_group):
                name = f"{rng.choice(PREFIXES)} {rng.choice(info.suffixes)}"
                merchants.append(
                    Merchant(f"M{len(merchants) + 1:04d}", name, category, country)
                )
    return merchants
