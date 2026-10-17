"""Every fault, by name."""

from __future__ import annotations

from forensics.faults import injectors as f

FAULTS: dict[str, f.Fault] = {
    fault.name: fault
    for fault in (
        f.TruncateDocument(),
        f.DropAmount(), f.TransposeAmount(), f.SwapCurrency(), f.ShiftDate(),
        f.InventParty(), f.InventAmount(), f.NumericDates(), f.SymbolCurrency(),
        f.MainFactsOnlyPrompt(),
        f.ConfusableLabel(), f.TitleOnlyPrompt(),
        f.SummaryInventsFee(), f.SummaryWrongAmount(), f.SummaryDropsAmounts(), f.EmbellishPrompt(),
    )
}


def get_fault(name: str) -> f.Fault:
    try:
        return FAULTS[name]
    except KeyError:
        raise KeyError(f"unknown fault {name!r}; known: {', '.join(FAULTS)}") from None
