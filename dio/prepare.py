"""Préparer les contextes horaires X et les courants y, sans entraîner de modèle.

    python -m dio.prepare --data-dir data --out data/prepared
    python -m dio.prepare --test-dir validation/test --schema data/prepared/schema.json --out /tmp/prepared_test

X utilise seulement les cinq contextes. Le schéma est appris sur annee1 puis
réutilisé : aucun courant, même initial, ne devient une colonne de X.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from make_validation import CONTEXTES, H


COLONNES = {
    "dmop": ["ut_ms", "subsystem"],
    "ftl": ["utb_ms", "ute_ms", "type", "flagcomms"],
    "evtf": ["ut_ms", "description"],
    "saaf": ["ut_ms", "sa", "sx", "sy", "sz"],
    "ltdata": ["ut_ms", "sunmars_km", "earthmars_km", "sunmarsearthangle_deg",
               "solarconstantmars", "eclipseduration_min", "occultationduration_min"],
}
ANGLES = COLONNES["saaf"][1:]
JOUR = 24 * H


def verifier_temps(temps, nom, unique=False):
    if temps.isna().any() or not pd.api.types.is_integer_dtype(temps):
        raise ValueError(f"{nom} : timestamps entiers en millisecondes requis")
    if not temps.is_monotonic_increasing:
        raise ValueError(f"{nom} : timestamps non triés")
    if unique and temps.duplicated().any():
        raise ValueError(f"{nom} : timestamps dupliqués")


def verifier_heures(heures):
    heures = pd.Index(heures, name="ut_ms")
    verifier_temps(pd.Series(heures), "grille horaire", unique=True)
    if len(heures) == 0 or (heures % H != 0).any() or (np.diff(heures) != H).any():
        raise ValueError("La grille doit contenir toutes les heures UTC consécutives")
    return heures


def lire_contextes(dossier):
    tables, bilan = {}, {}
    for nom in CONTEXTES:
        table = pd.read_csv(Path(dossier) / f"{nom}.csv")
        if table.columns.tolist() != COLONNES[nom]:
            raise ValueError(f"{nom} : colonnes inattendues")
        temps = "utb_ms" if nom == "ftl" else "ut_ms"
        # Un fichier vide avec son en-tête reste une source explicitement absente.
        if table.empty:
            table[temps] = table[temps].astype("int64")
        verifier_temps(table[temps], nom, unique=nom in ("saaf", "ltdata"))
        for colonne in table.select_dtypes(include="number"):
            if np.isinf(table[colonne]).any():
                raise ValueError(f"{nom}/{colonne} : valeur infinie")
        if nom == "ftl":
            if table.empty:
                table["ute_ms"] = table["ute_ms"].astype("int64")
            if not pd.api.types.is_integer_dtype(table["ute_ms"]):
                raise ValueError("ftl : fins entières requises")
            if table.isna().any().any() or (table.ute_ms <= table.utb_ms).any():
                raise ValueError("ftl : intervalle absent ou de durée non positive")
            if not table.flagcomms.isin([True, False]).all():
                raise ValueError("ftl : flagcomms doit être booléen")
        elif nom in ("dmop", "evtf") and table.isna().any().any():
            raise ValueError(f"{nom} : code absent")
        elif nom in ("saaf", "ltdata"):
            for colonne in COLONNES[nom][1:]:
                table[colonne] = pd.to_numeric(table[colonne], errors="raise")
            if nom == "ltdata" and (table.ut_ms % JOUR != 0).any():
                raise ValueError("ltdata : une ligne doit commencer à minuit UTC")
        tables[nom] = table
        bilan[nom] = {
            "lignes": len(table), "cellules_manquantes": int(table.isna().sum().sum()),
            "doublons_ligne": int(table.duplicated().sum()),
            "timestamps_repetes": int(table[temps].duplicated().sum()),
            "debut_ms": int(table[temps].min()) if len(table) else None,
            "fin_ms": int(table[temps].max()) if len(table) else None,
        }
    return tables, bilan


def cibles_horaires(chemin, colonnes=None):
    power = pd.read_csv(chemin)
    attendues = ["ut_ms"] + (colonnes if colonnes is not None else power.columns[1:].tolist())
    if power.columns.tolist() != attendues or power.empty:
        raise ValueError("power : colonnes inattendues ou fichier vide")
    verifier_temps(power.ut_ms, "power", unique=True)
    if not all(c.startswith("NPWD") for c in attendues[1:]):
        raise ValueError("power : circuits NPWD attendus")
    valeurs = power[attendues[1:]].to_numpy(dtype=float)
    if np.isinf(valeurs).any():
        raise ValueError("power : courant infini")
    # Même poids par mesure et même arrondi que le script officiel.
    power["ut_ms"] = (power["ut_ms"] // H) * H
    moyenne = power.groupby("ut_ms").mean().sort_index()
    heures = verifier_heures(range(int(moyenne.index.min()), int(moyenne.index.max()) + H, H))
    return moyenne.reindex(heures)


def decouper_intervalles(heures, debuts, fins):
    """Retourner (heure, ligne source, durée ms) pour chaque morceau d'intervalle."""
    debuts = np.maximum(np.asarray(debuts, dtype="int64"), heures[0])
    fins = np.minimum(np.asarray(fins, dtype="int64"), heures[-1] + H)
    lignes = np.flatnonzero(fins > debuts)
    if not len(lignes):
        vide = np.array([], dtype="int64")
        return vide, vide, vide
    premieres = debuts[lignes] // H * H
    indices, sources, durees = [], [], []
    # Boucler sur les heures traversées, pas sur les millions de mesures saaf.
    for decalage in range(int(((fins[lignes] - 1 - premieres) // H).max()) + 1):
        heures_morceaux = premieres + decalage * H
        duree = np.minimum(fins[lignes], heures_morceaux + H) - np.maximum(debuts[lignes], heures_morceaux)
        garder = duree > 0
        indices.append((heures_morceaux[garder] - heures[0]) // H)
        sources.append(lignes[garder])
        durees.append(duree[garder])
    return np.concatenate(indices), np.concatenate(sources), np.concatenate(durees)


def durees_union(heures, debuts, fins, minimum_actifs=1):
    """Durée en minutes avec au moins N lignes actives, sans double comptage."""
    variations = defaultdict(int)
    for debut, fin in zip(debuts, fins):
        variations[int(debut)] += 1
        variations[int(fin)] -= 1
    actifs, precedent = 0, None
    segments_debut, segments_fin = [], []
    for instant, variation in sorted(variations.items()):
        if precedent is not None and actifs >= minimum_actifs:
            segments_debut.append(precedent)
            segments_fin.append(instant)
        actifs += variation
        precedent = instant
    indices, _, durees = decouper_intervalles(heures, segments_debut, segments_fin)
    return np.bincount(indices, weights=durees, minlength=len(heures)) / 60_000


def compter(temps, heures):
    return (temps // H * H).value_counts().reindex(heures, fill_value=0).to_numpy()


def hors_plage(temps, heures):
    if temps.empty:
        return np.ones(len(heures), dtype=int)
    return ((heures + H <= temps.min()) | (heures > temps.max())).astype(int)


def codes_evenements(description):
    # Le séparateur _/_ introduit les paramètres observés RTLT, RA, RANGE ou SSP.
    codes = description.str.split("_/_", n=1).str[0]
    return codes.str.replace(r"((?:APOCENTRE|PERICENTRE)_PASSAGE)_\d+$", r"\1", regex=True)


def bornes_evenements(codes):
    marqueurs = codes.str.extract(r"(?:^|_)(START|END)(?:_|$)", expand=False)
    familles = codes.str.replace(r"(^|_)(START|END)(?=_|$)", "", regex=True).str.strip("_")
    return marqueurs, familles


def plus_frequents(codes, nombre):
    comptes = codes.value_counts().to_dict()
    return sorted(comptes, key=lambda code: (-comptes[code], code))[:nombre]


def apprendre_categories(tables, circuits):
    codes = codes_evenements(tables["evtf"].description)
    marqueurs, familles = bornes_evenements(codes)
    return {
        "version": 1,
        "periode_apprentissage": "annee1",
        "target_columns": circuits,
        "dmop_codes": plus_frequents(tables["dmop"].subsystem, 10),
        "ftl_codes": sorted(tables["ftl"].type.unique().tolist()),
        "evtf_intervals": plus_frequents(familles[marqueurs.eq("START")], 4),
        "saaf_hold_ms": 60_000,
    }


def preparer_ftl(table, heures, schema):
    x = pd.DataFrame(index=heures)
    # Ces codes simultanés ne forment pas une partition exclusive de l'heure.
    for code in schema["ftl_codes"]:
        lignes = table[table.type.eq(code)]
        x[f"ftl_{code}_minutes"] = durees_union(heures, lignes.utb_ms, lignes.ute_ms)
    autres = table[~table.type.isin(schema["ftl_codes"])]
    x["ftl_other_minutes"] = durees_union(heures, autres.utb_ms, autres.ute_ms)
    x["ftl_coverage_minutes"] = durees_union(heures, table.utb_ms, table.ute_ms)
    communications = table[table.flagcomms.eq(True)]
    # Au moins une ligne prévoit des communications, même si une autre vaut False.
    x["ftl_comms_minutes"] = durees_union(heures, communications.utb_ms, communications.ute_ms)
    x["ftl_overlap_minutes"] = durees_union(heures, table.utb_ms, table.ute_ms, minimum_actifs=2)
    return x


def preparer_dmop(table, heures, schema):
    x = pd.DataFrame(index=heures)
    for code in schema["dmop_codes"]:
        x[f"dmop_{code}_count"] = compter(table.loc[table.subsystem.eq(code), "ut_ms"], heures)
    autres = ~table.subsystem.isin(schema["dmop_codes"])
    x["dmop_other_count"] = compter(table.loc[autres, "ut_ms"], heures)
    x["dmop_total_count"] = compter(table.ut_ms, heures)
    temps = table.ut_ms.to_numpy(dtype="int64")
    # À h+1 exclue, une commande de l'heure suivante n'a pas encore eu lieu.
    positions = np.searchsorted(temps, heures + H, side="left") - 1
    ages = np.full(len(heures), np.nan)
    connues = positions >= 0
    ages[connues] = (heures[connues] + H - temps[positions[connues]]) / H
    x["dmop_hours_since_last"] = ages
    x["dmop_no_previous_command"] = (~connues).astype(int)
    x["dmop_outside_recorded_range"] = hors_plage(table.ut_ms, heures)
    return x


def apparier_evenements(temps, marqueurs, heures):
    """Conserver les paires certaines et marquer les portions ambiguës inconnues."""
    debuts, fins, inconnus_debut, inconnus_fin = [], [], [], []
    debut, ambigu, derniere_fin, anomalies = None, False, None, 0
    for instant, marqueur in zip(temps, marqueurs):
        instant = int(instant)
        if marqueur == "START":
            if debut is None:
                debut = instant
            else:
                ambigu = True
                anomalies += 1
        elif debut is None:
            inconnus_debut.append(heures[0] if derniere_fin is None else derniere_fin)
            inconnus_fin.append(instant)
            derniere_fin = instant
            anomalies += 1
        else:
            if ambigu or instant <= debut:
                inconnus_debut.append(debut)
                inconnus_fin.append(max(instant, debut + 1))
                if instant <= debut:
                    anomalies += 1
            else:
                debuts.append(debut)
                fins.append(instant)
            debut, ambigu, derniere_fin = None, False, instant
    if debut is not None:
        inconnus_debut.append(debut)
        inconnus_fin.append(heures[-1] + H)
        anomalies += 1
    durees = durees_union(heures, debuts, fins)
    inconnues = durees_union(heures, inconnus_debut, inconnus_fin) > 0
    durees[inconnues] = np.nan
    return durees, inconnues.astype(int), anomalies


def preparer_evtf(table, heures, schema):
    x = pd.DataFrame(index=heures)
    codes = codes_evenements(table.description)
    marqueurs, familles = bornes_evenements(codes)
    x["evtf_point_count"] = compter(table.loc[marqueurs.isna(), "ut_ms"], heures)
    x["evtf_start_count"] = compter(table.loc[marqueurs.eq("START"), "ut_ms"], heures)
    x["evtf_end_count"] = compter(table.loc[marqueurs.eq("END"), "ut_ms"], heures)
    for code in ("PERICENTRE_PASSAGE", "APOCENTRE_PASSAGE"):
        x[f"evtf_{code}_count"] = compter(table.loc[codes.eq(code), "ut_ms"], heures)
    autres = marqueurs.notna() & ~familles.isin(schema["evtf_intervals"])
    x["evtf_other_boundary_count"] = compter(table.loc[autres, "ut_ms"], heures)
    x["evtf_outside_recorded_range"] = hors_plage(table.ut_ms, heures)
    anomalies = {}
    for famille in schema["evtf_intervals"]:
        selection = familles.eq(famille) & marqueurs.notna()
        durees, inconnues, nombre = apparier_evenements(table.ut_ms[selection], marqueurs[selection], heures)
        if table.empty:
            durees[:] = np.nan
            inconnues[:] = 1
        x[f"evtf_{famille}_minutes"] = durees
        x[f"evtf_{famille}_unknown"] = inconnues
        anomalies[famille] = nombre
    return x, anomalies


def preparer_saaf(table, heures, schema):
    x = pd.DataFrame(index=heures)
    temps = table.ut_ms.to_numpy(dtype="int64")
    limite = schema["saaf_hold_ms"]
    fins = np.minimum(np.r_[temps[1:], temps[-1:] + limite], temps + limite)
    indices, sources, durees = decouper_intervalles(heures, temps, fins)
    x["saaf_coverage_fraction"] = np.bincount(indices, weights=durees, minlength=len(heures)) / H
    for angle in ANGLES:
        valeurs = table[angle].to_numpy(dtype=float)[sources]
        valides = np.isfinite(valeurs)
        poids = np.bincount(indices[valides], weights=durees[valides], minlength=len(heures))
        sommes = np.bincount(indices[valides], weights=valeurs[valides] * durees[valides], minlength=len(heures))
        moyenne = np.full(len(heures), np.nan)
        np.divide(sommes, poids, out=moyenne, where=poids > 0)
        x[f"saaf_{angle}_mean"] = moyenne
        extremes = pd.Series(valeurs, index=indices).groupby(level=0).agg(["min", "max"])
        amplitude = (extremes["max"] - extremes["min"]).reindex(range(len(heures)))
        x[f"saaf_{angle}_range"] = amplitude.to_numpy()
    x["saaf_missing"] = x.isna().any(axis=1).astype(int)
    return x


def preparer_ltdata(table, heures):
    jours = heures // JOUR * JOUR
    x = table.set_index("ut_ms").reindex(jours)
    x.index = heures
    x = x.add_prefix("ltdata_")
    x["ltdata_missing"] = x.isna().any(axis=1).astype(int)
    return x


def construire_entrees(tables, heures, schema):
    """X brut : cinq contextes, aucune consommation ni statistique de consommation."""
    heures = verifier_heures(heures)
    evenements, anomalies = preparer_evtf(tables["evtf"], heures, schema)
    x = pd.concat([
        preparer_ftl(tables["ftl"], heures, schema),
        preparer_dmop(tables["dmop"], heures, schema),
        evenements,
        preparer_saaf(tables["saaf"], heures, schema),
        preparer_ltdata(tables["ltdata"], heures),
    ], axis=1)
    if x.columns.duplicated().any() or np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError("X : colonnes dupliquées ou valeur infinie")
    return x, anomalies


def remplir_entrees(x, schema):
    if x.columns.tolist() != schema["feature_columns"]:
        raise ValueError("X : colonnes ou ordre différents du schéma d'annee1")
    x = x.fillna(schema["fill_values"])
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("X : il reste des valeurs manquantes ou infinies")
    return x


def enregistrer(dossier, x_brut, schema, bilan_sources, anomalies, y=None, nom_cible="y.csv"):
    x = remplir_entrees(x_brut, schema)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    x.to_csv(dossier / "X.csv")
    if y is not None:
        y.to_csv(dossier / nom_cible)
    bilan = {
        "X_dimensions_csv": [len(x), len(x.columns) + 1],
        "X_debut_ms": int(x.index[0]), "X_fin_ms": int(x.index[-1]),
        "X_cellules_remplies": int(x_brut.isna().sum().sum()),
        "X_manquants_avant_remplissage": {c: int(n) for c, n in x_brut.isna().sum().items() if n},
        "X_manquants_apres_remplissage": int(x.isna().sum().sum()),
        "X_infinis": int(np.isinf(x.to_numpy(dtype=float)).sum()),
        "X_timestamps_dupliques": int(x.index.duplicated().sum()),
        "sources": bilan_sources,
        "evtf_paires_anormales": anomalies,
    }
    if y is not None:
        bilan["cibles"] = {
            "fichier": nom_cible, "dimensions_csv": [len(y), len(y.columns) + 1],
            "heures_sans_courant": int(y.isna().all(axis=1).sum()),
            "cellules_inconnues": int(y.isna().sum().sum()),
        }
    (dossier / "bilan.json").write_text(json.dumps(bilan, indent=2, allow_nan=False) + "\n")
    print(f"{dossier} : X {len(x)} x {len(x.columns) + 1}, {bilan['X_cellules_remplies']} cellules remplies")
    return bilan


def preparer_periodes(data_dir, out):
    out = Path(out)
    schema = None
    for periode in ("annee1", "annee2"):
        dossier = Path(data_dir) / periode
        y = cibles_horaires(dossier / "power.csv", None if schema is None else schema["target_columns"])
        tables, bilan = lire_contextes(dossier)
        if schema is None:
            if len(y.columns) != 33:
                raise ValueError("33 circuits sont requis")
            schema = apprendre_categories(tables, y.columns.tolist())
        x, anomalies = construire_entrees(tables, y.index, schema)
        if periode == "annee1":
            schema["feature_columns"] = x.columns.tolist()
            medianes = x.median()
            if medianes.isna().any():
                raise ValueError("Impossible d'apprendre un remplissage : colonne entièrement inconnue sur annee1")
            schema["fill_values"] = medianes.to_dict()
            out.mkdir(parents=True, exist_ok=True)
            (out / "schema.json").write_text(json.dumps(schema, indent=2, allow_nan=False) + "\n")
        enregistrer(out / periode, x, schema, bilan, anomalies, y)


def preparer_test(test_dir, schema_path, out):
    dossier = Path(test_dir)
    schema = json.loads(Path(schema_path).read_text())
    heures_csv = pd.read_csv(dossier / "heures_a_predire.csv")
    if heures_csv.columns.tolist() != ["ut_ms"]:
        raise ValueError("heures_a_predire : seule la colonne ut_ms est attendue")
    heures = verifier_heures(heures_csv.ut_ms)
    tables, bilan = lire_contextes(dossier)
    x, anomalies = construire_entrees(tables, heures, schema)
    y_init = cibles_horaires(dossier / "power_init.csv", schema["target_columns"])
    if y_init.index[-1] >= heures[0]:
        raise ValueError("power_init empiète sur les heures à prédire")
    return enregistrer(out, x, schema, bilan, anomalies, y_init, "y_init.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    groupe = ap.add_mutually_exclusive_group(required=True)
    groupe.add_argument("--data-dir", type=Path)
    groupe.add_argument("--test-dir", type=Path)
    ap.add_argument("--schema", type=Path, help="schéma produit uniquement depuis annee1")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.test_dir:
        if args.schema is None:
            ap.error("--schema est requis avec --test-dir")
        preparer_test(args.test_dir, args.schema, args.out)
    else:
        if args.schema is not None:
            ap.error("--schema est réservé au mode test")
        preparer_periodes(args.data_dir, args.out)


if __name__ == "__main__":
    main()
