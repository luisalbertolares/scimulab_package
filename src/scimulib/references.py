"""Declarative validation of the references used in energy calculations.

A `ReferenceSpec` states *what* a valid group of references looks like, as three
ordered tuples of small rules. `ReferenceSpec.resolve` and `ReferenceSpec.store`
are the only places that decide *when* each tuple runs:

    key rules     on the reference names, before any value is read
    entity rules  once per entity reference
    set rules     on the whole group, after the energies are read

A new check is a function listed in the matching tuple: `(Entry) -> None` for an
entity rule and `(Refs) -> None` for a key or set rule. Each rule raises its own
exception, and the order of a tuple is the order in which its checks run.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING

from scimulib.exceptions import (
    EnergyCalculationError,
    MismatchedGeometryError,
    MissingGeometryError,
    MissingReferenceError,
    StoichiometryError,
)

if TYPE_CHECKING:
    from scimulib.scimulib import ChemicalEntity

    Reference = ChemicalEntity | float


def _is_entity(obj: object) -> bool:
    """Return whether `obj` is a `ChemicalEntity` without importing it eagerly."""
    # Deferred import: scimulib.scimulib imports this module.
    from scimulib.scimulib import ChemicalEntity

    return isinstance(obj, ChemicalEntity)


class Entry:
    """One entity reference together with what an entity rule needs to judge it."""

    def __init__(self, target: ChemicalEntity, key: str, ref: ChemicalEntity, label: str):
        """
        Args:
            target: Entity the reference is validated against.
            key: Name the reference was given.
            ref: The entity reference itself.
            label: Kind of reference, used in error messages.
        """
        self.target = target
        self.key = key
        self.ref = ref
        self.label = label

    @cached_property
    def composition(self) -> dict[str, float]:
        """Composition of the reference entity, keyed by species."""
        return self.ref.composition

    @cached_property
    def atoms(self) -> float:
        """Total number of atoms in the reference entity."""
        return sum(self.composition.values())


class Refs:
    """The references of one calculation, plus the context the set rules need."""

    def __init__(
        self,
        target: ChemicalEntity,
        label: str,
        stored: Mapping[str, Reference],
        named: Mapping[str, Reference],
        coefficients: Mapping[str, float] | None = None,
        require_complete: bool = True,
    ):
        """
        Args:
            target: Entity the references are validated against.
            label: Kind of reference, used in error messages.
            stored: References kept by the entity.
            named: References supplied for this call; they override stored ones.
            coefficients: Optional weights keyed like the references.
            require_complete: Whether the references must cover the target.
        """
        self.target = target
        self.label = label
        self.stored = stored
        self.named = named
        self.coefficients = coefficients
        self.require_complete = require_complete

    @cached_property
    def target_symbols(self) -> tuple[str, ...]:
        """Species of the target, read once per resolution."""
        return tuple(self.target.symbols)

    @cached_property
    def merged(self) -> dict[str, Reference]:
        """Stored references, overridden by the ones named in this call."""
        return {**self.stored, **self.named}

    @cached_property
    def entries(self) -> dict[str, Entry]:
        """Entity references keyed by their reference names."""
        return {
            key: Entry(self.target, key, ref, self.label)
            for key, ref in self.merged.items()
            if _is_entity(ref)
        }

    @property
    def all_entities(self) -> bool:
        """Whether every merged reference is a `ChemicalEntity`."""
        return len(self.entries) == len(self.merged)

    @cached_property
    def covered_species(self) -> set[str]:
        """Species represented by the entity references in this group."""
        return {species for entry in self.entries.values() for species in entry.composition}

    @cached_property
    def energies(self) -> dict[str, float]:
        """Energy of every reference; entities must carry a total energy."""
        return {
            key: self.entries[key].ref.total_energy if key in self.entries else float(ref)
            for key, ref in self.merged.items()
        }


EntityRule = Callable[[Entry], None]
SetRule = Callable[[Refs], None]
Selector = Callable[[Refs, dict[str, float]], dict[str, float]]


class ReferenceSpec:
    """What a valid group of references looks like."""

    def __init__(
        self,
        label: str,
        key_rules: tuple[SetRule, ...] = (),
        entity_rules: tuple[EntityRule, ...] = (),
        set_rules: tuple[SetRule, ...] = (),
        select: Selector | None = None,
    ):
        """
        Args:
            label: Names this kind of reference in error messages.
            key_rules: Checks on the reference names, run before any value is read.
            entity_rules: Checks run on every entity reference, in order.
            set_rules: Checks on the whole group, run once the energies are read.
            select: Optional filter that picks the energies a calculation uses.
        """
        self.label = label
        self.key_rules = tuple(key_rules)
        self.entity_rules = tuple(entity_rules)
        self.set_rules = tuple(set_rules)
        self.select = select

    def extended(self, label: str, *entity_rules: EntityRule) -> ReferenceSpec:
        """Copy of this spec under a new label, with more entity rules run last."""
        return ReferenceSpec(
            label, self.key_rules, self.entity_rules + entity_rules, self.set_rules, self.select
        )

    def resolve(
        self,
        target: ChemicalEntity,
        named: Mapping[str, Reference],
        *,
        stored: Mapping[str, Reference] | None = None,
        coefficients: Mapping[str, float] | None = None,
        require_complete: bool = True,
    ) -> dict[str, float]:
        """Validate stored and per-call references and return their energies.

        Args:
            target: Entity the references are validated against.
            named: References supplied for this call; they override stored ones.
            stored: Defaults kept by the entity, revalidated on every call.
            coefficients: Optional positive weights keyed like the references.
            require_complete: Whether the references must cover the target.

        Raises:
            ValueError: If the coefficient keys differ from the reference keys.
            StoichiometryError: If a rule or a coefficient rejects the references.
            MissingReferenceError: If a required reference is missing.
            MissingDataError: If an entity reference has no total energy.
            EnergyCalculationError: If a returned energy is not finite.
        """
        refs = Refs(
            target, self.label, stored if stored is not None else {}, named,
            coefficients, require_complete,
        )
        for rule in self.key_rules:
            rule(refs)
        for entry in refs.entries.values():
            for rule in self.entity_rules:
                rule(entry)
        energies = refs.energies
        _check_coefficients(coefficients, energies)
        for rule in self.set_rules:
            rule(refs)
        selected = self.select(refs, energies) if self.select is not None else energies
        _check_finite(selected)
        return selected

    def store(
        self, target: ChemicalEntity, references: Mapping[str, Reference] | None
    ) -> dict[str, Reference]:
        """Validate references kept by an entity at construction and copy them."""
        if references is None:
            return {}
        if not isinstance(references, Mapping):
            raise TypeError(f"{self.label.capitalize()} references must be a mapping.")
        stored = dict(references)
        if stored:
            self.resolve(target, {}, stored=stored, require_complete=False)
        return stored


def _check_coefficients(
    coefficients: Mapping[str, float] | None, energies: Mapping[str, float]
) -> None:
    """Validate optional reference weights against the resolved energies.

    Args:
        coefficients: Optional positive, finite weights keyed by reference name.
        energies: Resolved reference energies, used to determine the required keys.

    Raises:
        ValueError: If the coefficient keys do not match the energy keys.
        StoichiometryError: If any coefficient is non-finite or not positive.
    """
    if coefficients is None:
        return
    if coefficients.keys() != energies.keys():
        raise ValueError("Stoichiometric coefficients must match the reference keys.")
    for key, coefficient in coefficients.items():
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise StoichiometryError(
                f"Stoichiometric coefficient for '{key}' must be positive "
                f"and finite, got {coefficient}."
            )


def _check_finite(energies: Mapping[str, float]) -> None:
    """Require every resolved reference energy to be finite.

    Args:
        energies: Reference energies keyed by reference name.

    Raises:
        EnergyCalculationError: If any energy is not finite.
    """
    for key, energy in energies.items():
        if not math.isfinite(energy):
            raise EnergyCalculationError(
                f"Reference energy for '{key}' must be finite, got {energy}."
            )


# --- Key rules ---

def stored_keys_in_target(refs: Refs) -> None:
    """Stored references keyed by element may only name species of the target."""
    stored_only = refs.stored.keys() - refs.named.keys()
    extra = stored_only - set(refs.target_symbols) if stored_only else set()
    if extra:
        raise StoichiometryError(
            f"Stored {refs.label} references contain species not present in "
            f"{refs.target.name}: {sorted(extra)}."
        )


def keys_within(required: Callable[[ChemicalEntity], set[str]]) -> SetRule:
    """Reference names must be among the species `required(target)`."""
    def rule(refs: Refs) -> None:
        """Reject reference names outside the required species."""
        extra = set(refs.merged) - required(refs.target)
        if extra:
            raise StoichiometryError(
                f"Unknown {refs.label} reference species: {sorted(extra)}."
            )
    return rule


def keys_cover(required: Callable[[ChemicalEntity], set[str]]) -> SetRule:
    """Reference names must include every species `required(target)`."""
    def rule(refs: Refs) -> None:
        """Reject missing required names when complete coverage is required."""
        missing = required(refs.target) - set(refs.merged)
        if missing and refs.require_complete:
            raise MissingReferenceError(
                f"Missing {refs.label} references: {sorted(missing)}.",
                missing_species=missing,
            )
    return rule


# --- Entity rules ---

def has_geometry(entry: Entry) -> None:
    """The reference must have atoms (reading its composition needs a geometry)."""
    if not entry.composition:
        raise MissingGeometryError(
            f"{entry.label.capitalize()} reference '{entry.key}' has no atoms."
        )


def single_species(entry: Entry) -> None:
    """Require an entity reference to contain exactly one species.

    Raises:
        StoichiometryError: If the reference composition has zero or multiple species.
    """
    if len(entry.composition) != 1:
        raise StoichiometryError(
            f"{entry.label.capitalize()} references must represent a single "
            f"species, got: {list(entry.composition)}."
        )


def key_in_composition(entry: Entry) -> None:
    """Require the reference name to appear in the reference composition.

    Raises:
        StoichiometryError: If the named species is absent from the composition.
    """
    if entry.key not in entry.composition:
        raise StoichiometryError(
            f"Reference '{entry.key}' must contain species '{entry.key}', "
            f"got {list(entry.composition)}."
        )


def atom_count(expected: int) -> EntityRule:
    """The reference must have exactly `expected` atoms."""
    def rule(entry: Entry) -> None:
        """Reject an entity reference whose atom count differs from `expected`."""
        if entry.atoms != expected:
            raise StoichiometryError(
                f"{entry.label.capitalize()} references must have {expected} "
                f"atom(s), got: {entry.atoms:g}."
            )
    return rule


def same_atom_count(entry: Entry) -> None:
    """Require the reference and target to have the same total atom count.

    Raises:
        StoichiometryError: If their total atom counts differ.
    """
    expected = sum(entry.target.composition.values())
    if entry.atoms != expected:
        raise StoichiometryError(
            f"{entry.label.capitalize()} references must match the number of atoms "
            f"({expected:g}), got: {entry.atoms:g}."
        )


def same_geometry(entry: Entry) -> None:
    """Atomic positions must match the target's, ignoring species."""
    if not entry.target.has_same_geometry(entry.ref, ignore_species=True):
        raise MismatchedGeometryError(
            f"{entry.label.capitalize()} references must share the same geometry "
            f"as {entry.target.name}."
        )


# --- Set rules ---

def keys_cover_target(refs: Refs) -> None:
    """Element-keyed references must cover every species of the target."""
    missing = set(refs.target_symbols) - set(refs.merged)
    if missing and refs.require_complete:
        raise MissingReferenceError(
            f"{refs.label.capitalize()} references must cover all species in "
            f"{refs.target.name}. Missing species: {sorted(missing)}.",
            missing_species=missing,
        )


def species_cover_target(refs: Refs) -> None:
    """Entity references, taken together, must contain every species of the target."""
    if not refs.all_entities:
        return
    missing = set(refs.target_symbols) - refs.covered_species
    if missing and refs.require_complete:
        raise MissingReferenceError(
            f"{refs.label.capitalize()} references must cover all species in "
            f"{refs.target.name}. Missing species: {sorted(missing)}.",
            missing_species=missing,
        )


def species_within_target(refs: Refs) -> None:
    """Entity references may not introduce species absent from the target."""
    if not refs.all_entities:
        return
    extra = refs.covered_species - set(refs.target_symbols)
    if extra:
        raise StoichiometryError(
            f"{refs.label.capitalize()} references contain species not present in "
            f"{refs.target.name}: {sorted(extra)}."
        )


def mass_balance(refs: Refs) -> None:
    """Weighted entity references must add up to the composition of the target."""
    if refs.all_entities:
        check_mass_balance(
            refs.target,
            {key: entry.ref for key, entry in refs.entries.items()},
            refs.label,
            refs.coefficients,
        )


# --- Selectors ---

def select_target_species(refs: Refs, energies: dict[str, float]) -> dict[str, float]:
    """Keep only the energies of species present in the target."""
    return {symbol: energies[symbol] for symbol in refs.target_symbols if symbol in energies}


# --- Stand-alone checks ---

def check_mass_balance(
    target: ChemicalEntity,
    parts: Mapping[str, ChemicalEntity],
    label: str = "system",
    coefficients: Mapping[str, float] | None = None,
) -> None:
    """Require `sum(coefficients[k] * parts[k]) == target`, atom by atom."""
    combined: Counter[str] = Counter()
    for key, part in parts.items():
        weight = 1 if coefficients is None else coefficients[key]
        for species, amount in part.composition.items():
            combined[species] += weight * amount
    expected = target.composition
    if combined.keys() != expected.keys() or any(
        not math.isclose(combined[species], amount, rel_tol=0, abs_tol=1e-8)
        for species, amount in expected.items()
    ):
        raise StoichiometryError(
            f"Mass balance mismatch for {label} references. "
            f"Combined fragments: {dict(combined)}, expected: {dict(expected)}."
        )


# --- Specs shared by every entity ---

ATOMIC = ReferenceSpec(
    "atom",
    key_rules=(stored_keys_in_target,),
    entity_rules=(single_species, key_in_composition, atom_count(1)),
    set_rules=(keys_cover_target,),
    select=select_target_species,
)

MASS_BALANCE = ReferenceSpec(
    "system",
    entity_rules=(has_geometry,),
    set_rules=(species_cover_target, species_within_target, mass_balance),
)
