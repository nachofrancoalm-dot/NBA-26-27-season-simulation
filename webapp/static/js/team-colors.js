// team-colors.js -- colores oficiales de marca de las 30 franquicias,
// tabla estática indexada por abreviatura, aplicada dinámicamente según
// status.team.abbreviation (mantiene "cualquier equipo cambiando el
// YAML" también en el frontend). [primary, secondary] en hex -- tonos de
// referencia de uso común, no valores certificados al pixel.
export const TEAM_COLORS = {
  ATL: ["#E03A3E", "#26282A"],
  BOS: ["#007A33", "#BA9653"],
  BKN: ["#000000", "#FFFFFF"],
  CHA: ["#1D1160", "#00788C"],
  CHI: ["#CE1141", "#000000"],
  CLE: ["#860038", "#041E42"],
  DAL: ["#00538C", "#002B5E"],
  DEN: ["#0E2240", "#FEC524"],
  DET: ["#C8102E", "#1D42BA"],
  GSW: ["#1D428A", "#FFC72C"],
  HOU: ["#CE1141", "#000000"],
  IND: ["#002D62", "#FDBB30"],
  LAC: ["#C8102E", "#1D428A"],
  LAL: ["#552583", "#FDB927"],
  MEM: ["#5D76A9", "#12173F"],
  MIA: ["#98002E", "#F9A01B"],
  MIL: ["#00471B", "#EEE1C6"],
  MIN: ["#0C2340", "#236192"],
  NOP: ["#0C2340", "#C8102E"],
  NYK: ["#006BB6", "#F58426"],
  OKC: ["#007AC1", "#EF3B24"],
  ORL: ["#0077C0", "#C4CED4"],
  PHI: ["#006BB6", "#ED174C"],
  PHX: ["#1D1160", "#E56020"],
  POR: ["#E03A3E", "#000000"],
  SAC: ["#5A2D81", "#63727A"],
  SAS: ["#8A8D8F", "#000000"],
  TOR: ["#CE1141", "#000000"],
  UTA: ["#002B5C", "#F9A01B"],
  WAS: ["#002B5C", "#E31837"],
};

/** Aplica [primary, secondary] como variables CSS globales
 * (--team-primary / --team-secondary), leídas por hero.css. Si no está
 * en la tabla, se deja el fallback de tokens.css. */
export function applyTeamColors(abbreviation) {
  const colors = TEAM_COLORS[abbreviation];
  if (!colors) return;
  const root = document.documentElement.style;
  root.setProperty("--team-primary", colors[0]);
  root.setProperty("--team-secondary", colors[1]);
}
