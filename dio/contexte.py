"""Variables horaires tirées des contextes bruts (saaf, ltdata, evtf), au lieu du X de prepare.py.

Expériences de docs/JOURNAL_RECETTE.md (points 5 et 6). Aucune consommation n'est lue.
- soleil : cosinus de chaque mesure saaf avant la moyenne horaire (deux angles 0° et 90° :
  moyenne des cosinus 0.5, cosinus de l'angle moyen 0.707), moyennes des angles, écart-type de sa.
  Heures sans mesure : NaN, puis remplissage choisi par l'appelant ;
- ltdata_lineaire : valeurs journalières (minuit UTC) interpolées au milieu de chaque heure ;
- geometrie : péricentres aux instants exacts, lus au milieu de l'heure ; altitude approchée ;
- ombres : fractions horaires d'ombre des lunes et d'occultation ;
- contacts : fraction de l'heure avec au moins une station (union) et somme des couvertures par station.
"""
import numpy as np
import pandas as pd

from . import prepare

H = 3_600_000
ANGLES = ("sa", "sx", "sy", "sz")
ALTITUDE_PERICENTRE_KM, ALTITUDE_APOCENTRE_KM = 330.0, 10_100.0
SOUS_POINTS = 4
DUREE_MAX_OMBRE_MS = 6 * H     # ombres et occultations : quelques minutes à une heure
DUREE_MAX_CONTACT_MS = 24 * H  # un passage de station dure moins d'un jour
OMBRES = ("PHO_UMBRA", "PHO_PENUMBRA", "DEI_UMBRA", "DEI_PENUMBRA", "OCC_PHOBOS", "OCC_DEIMOS", "OCC_MARS",
          "OCC_MARS_200KM")


def soleil(saaf, heures, maintien_ms=60_000):
    """Moyennes horaires pondérées par la durée (même découpage que prepare.preparer_saaf) des angles,
    de leurs cosinus, et écart-type de sa. NaN quand l'heure n'a aucune mesure."""
    heures = np.asarray(heures, dtype=np.int64)
    temps = saaf.ut_ms.to_numpy(dtype=np.int64)
    fins = np.minimum(np.r_[temps[1:], temps[-1:] + maintien_ms], temps + maintien_ms)
    indices, sources, durees = prepare.decouper_intervalles(heures, temps, fins)
    poids = np.bincount(indices, weights=durees, minlength=len(heures))

    def moyenne(valeurs):
        sortie = np.full(len(heures), np.nan)
        np.divide(np.bincount(indices, weights=valeurs[sources] * durees, minlength=len(heures)), poids,
                  out=sortie, where=poids > 0)
        return sortie

    sortie = {}
    for a in ANGLES:
        angle = saaf[a].to_numpy(dtype=np.float64)
        sortie[f"saaf_{a}_mean"] = moyenne(angle)
        sortie[f"saaf_{a}_cos"] = moyenne(np.cos(np.deg2rad(angle)))
    sa = saaf["sa"].to_numpy(dtype=np.float64)
    sortie["saaf_sa_ecart"] = np.sqrt(np.maximum(moyenne(sa ** 2) - sortie["saaf_sa_mean"] ** 2, 0))
    return pd.DataFrame(sortie, index=heures)


def interpoler(table):
    """Interpolation linéaire dans le temps des trous, bords remplis par la valeur la plus proche."""
    return table.interpolate(method="index", limit_direction="both")


def ltdata_lineaire(ltdata, heures):
    """Colonnes ltdata interpolées linéairement au milieu de chaque heure (bords : valeur la plus proche)."""
    milieux = np.asarray(heures, dtype=np.float64) + H / 2
    jours = ltdata.ut_ms.to_numpy(dtype=np.float64)
    return pd.DataFrame({f"ltdata_{c}": np.interp(milieux, jours, ltdata[c].to_numpy(dtype=np.float64))
                         for c in ltdata.columns if c != "ut_ms"}, index=heures)


def codes(evtf):
    return prepare.codes_evenements(evtf.description)


def geometrie(evtf, heures):
    """Au milieu de chaque heure : heures depuis le péricentre précédent et jusqu'au suivant, période,
    phase en sin/cos ; altitude approchée (moyenne de SOUS_POINTS sous-points, log interpolé)."""
    heures = np.asarray(heures, dtype=np.int64)
    c = codes(evtf)
    peri = np.sort(evtf.ut_ms[c == "PERICENTRE_PASSAGE"].to_numpy(dtype=np.float64))
    periode_mediane = np.median(np.diff(peri))
    peri = np.r_[peri[0] - periode_mediane, peri, peri[-1] + periode_mediane]  # bords : une orbite de plus
    milieux = heures + H / 2
    k = np.clip(np.searchsorted(peri, milieux, side="right") - 1, 0, len(peri) - 2)
    depuis, periode = milieux - peri[k], peri[k + 1] - peri[k]
    phase = 2 * np.pi * depuis / periode
    sortie = {"geo_pericentre_depuis_h": depuis / H, "geo_pericentre_jusqua_h": (peri[k + 1] - milieux) / H,
              "geo_periode_h": periode / H, "geo_phase_sin": np.sin(phase), "geo_phase_cos": np.cos(phase)}
    # Altitude : péricentre, apocentre et passages à N km (montée ou descente).
    altitudes = c.str.extract(r"^(\d+)_KM_(?:ASCEND|DESCEND)$", expand=False).astype(float)
    altitudes[c == "PERICENTRE_PASSAGE"] = ALTITUDE_PERICENTRE_KM
    altitudes[c == "APOCENTRE_PASSAGE"] = ALTITUDE_APOCENTRE_KM
    connus = altitudes.notna().to_numpy()
    points = pd.Series(np.log(altitudes[connus].to_numpy()), index=evtf.ut_ms[connus].to_numpy())
    points = points.groupby(level=0).mean()
    sous = heures[:, None] + (np.arange(SOUS_POINTS) + 0.5) * H / SOUS_POINTS
    log_alt = np.interp(sous.ravel(), points.index.to_numpy(dtype=np.float64), points.to_numpy()).reshape(sous.shape)
    sortie["geo_altitude_km"] = np.exp(log_alt).mean(axis=1)
    return pd.DataFrame(sortie, index=heures)


def intervalles(temps, marqueurs, ouverture, fermeture, duree_max_ms):
    """Paires (début, fin) d'une suite d'événements : ouverture … fermeture ; ouverture répétée ignorée,
    fermeture sans ouverture ignorée. Une paire plus longue que duree_max_ms est écartée : c'est une
    ouverture dont la fermeture manque (ex. DEI_UMBRA_END rangé avant DEI_UMBRA_START au même instant)."""
    debuts, fins, debut = [], [], None
    for t, m in zip(temps, marqueurs):
        if m == ouverture and debut is None:
            debut = t
        elif m == fermeture and debut is not None:
            if t - debut <= duree_max_ms:
                debuts.append(debut)
                fins.append(t)
            debut = None
    return np.array(debuts, dtype=np.int64), np.array(fins, dtype=np.int64)


def ombres(evtf, heures):
    """Fraction de chaque heure dans chaque intervalle OMBRES (paires START / END)."""
    heures = np.asarray(heures, dtype=np.int64)
    c = codes(evtf)
    sortie = {}
    for nom in OMBRES:
        lignes = c.str.fullmatch(f"{nom}_(?:START|END)")
        d, f = intervalles(evtf.ut_ms[lignes].to_numpy(), c[lignes].str.rsplit("_", n=1).str[1].to_numpy(),
                           "START", "END", DUREE_MAX_OMBRE_MS)
        sortie[f"geo_{nom.lower()}_fraction"] = prepare.durees_union(heures, d, f) / 60
    return pd.DataFrame(sortie, index=heures)


def contacts(evtf, heures):
    """Stations sol (paires XXX_AOS_10 / XXX_LOS_10 : seuil d'élévation de 10°, le seul présent pour
    toutes les stations des deux années) : fraction de l'heure avec au moins une station (union) et somme
    des fractions par station (peut dépasser 1 quand des stations se chevauchent)."""
    heures = np.asarray(heures, dtype=np.int64)
    c = codes(evtf)
    stations = sorted(set(c.str.extract(r"^([A-Z]{3})_(?:AOS|LOS)_10$", expand=False).dropna()))
    debuts, fins, somme = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)], np.zeros(len(heures))
    for s in stations:
        lignes = c.str.fullmatch(f"{s}_(?:AOS|LOS)_10")
        d, f = intervalles(evtf.ut_ms[lignes].to_numpy(), c[lignes].str[4:7].to_numpy(), "AOS", "LOS",
                           DUREE_MAX_CONTACT_MS)
        somme += prepare.durees_union(heures, d, f) / 60
        debuts.append(d)
        fins.append(f)
    union = prepare.durees_union(heures, np.concatenate(debuts), np.concatenate(fins)) / 60
    return pd.DataFrame({"geo_contact_union": union, "geo_contact_somme": somme}, index=heures)
