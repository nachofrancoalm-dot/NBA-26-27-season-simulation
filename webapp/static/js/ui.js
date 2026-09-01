// ui.js -- helpers de renderizado compartidos entre vistas (tablas, stat
// tiles, glosario, estado vacío, insignia de equipo).

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Tooltip compartido (#chart-tooltip en index.html) usado por charts.js,
 * court.js y leaderboard.js. */
export function showTooltip(event, text) {
  if (!text) return;
  const tip = document.getElementById("chart-tooltip");
  tip.textContent = text;
  tip.style.left = `${event.clientX}px`;
  tip.style.top = `${event.clientY}px`;
  tip.classList.add("visible");
}

export function hideTooltip() {
  document.getElementById("chart-tooltip").classList.remove("visible");
}

export function clear(container) {
  container.replaceChildren();
}

export function emptyState(message) {
  return el("div", { class: "empty-state" }, message);
}

export function apiErrorState(err) {
  return emptyState(err.message || "No se encontró el dataset todavía.");
}

/** Placeholder de carga con shimmer. `lines` describe el contenido
 * aproximado que va a llegar, para minimizar el salto de layout. */
export function skeleton(lines = ["title", "", "short"]) {
  return el(
    "div",
    { class: "card", "aria-hidden": "true" },
    lines.map((variant) => el("div", { class: `skeleton-line ${variant}`.trim() }))
  );
}

export function statTile(label, value) {
  return el("div", { class: "stat-tile" }, [
    el("p", { class: "label" }, label),
    el("p", { class: "value" }, value),
  ]);
}

export function statGrid(items) {
  return el(
    "div",
    { class: "grid" },
    items.map(([label, value]) => statTile(label, value))
  );
}

/** Selector compacto tipo pill, control LOCAL junto a cada tabla.
 * `options`: [{value, label}]. Llama a `onChange(value)` al cambiar. */
export function pillToggle(options, current, onChange) {
  const container = el("div", { class: "segmented" });
  const buttons = options.map((opt) =>
    el(
      "button",
      {
        type: "button",
        "aria-pressed": String(opt.value === current),
        onclick: () => {
          buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i].value === opt.value)));
          onChange(opt.value);
        },
      },
      opt.label
    )
  );
  container.append(...buttons);
  return container;
}

/** Insignia genérica: iniciales de respaldo + intento de imagen cargada
 * EN VIVO (nunca guardada en el repo -- ver README). Texto e imagen son
 * ALTERNATIVOS, nunca se apilan (si quedan los dos en el DOM se ve como
 * "logos con letras a la izquierda"). Usado por teamBadge() y la foto
 * del popup de jugador. */
export function photoBadge(src, fallbackText, size = 40, extraClass = "") {
  const label = document.createTextNode(fallbackText || "?");
  const badge = el("div", { class: `team-badge ${extraClass}`.trim() }, [label]);
  badge.style.width = `${size}px`;
  badge.style.height = `${size}px`;
  badge.style.fontSize = `${Math.round(size * 0.34)}px`;
  if (src) {
    const img = el("img", {
      src,
      alt: fallbackText || "",
      onerror: (event) => event.target.remove(),
      onload: () => label.remove(),
    });
    badge.append(img);
  }
  return badge;
}

/** Insignia de equipo -- caso particular de photoBadge() con la URL de
 * cdn.nba.com ya armada a partir del team_id. */
export function teamBadge(teamId, abbreviation, size = 40) {
  const src = teamId ? `https://cdn.nba.com/logos/nba/${teamId}/global/L/logo.svg` : null;
  return photoBadge(src, abbreviation ? abbreviation.slice(0, 3) : "NBA", size);
}

/** Foto de jugador vía cdn.nba.com/headshots -- cae a iniciales si no carga. */
export function playerPhoto(playerId, playerName, size = 96) {
  const initials = (playerName || "?")
    .split(" ")
    .map((part) => part[0])
    .filter(Boolean)
    .slice(0, 2)
    .join("");
  const src = playerId ? `https://cdn.nba.com/headshots/nba/latest/1040x760/${playerId}.png` : null;
  return photoBadge(src, initials, size, "player-photo");
}

/** Celda de tabla: insignia pequeña + abreviatura, para usar como
 * formatter de dataTable() en columnas de equipo (p.ej. standings). */
export function teamCell(teamId, abbreviation) {
  return el("div", { style: "display: flex; align-items: center; gap: 8px;" }, [
    teamBadge(teamId, abbreviation, 22),
    abbreviation,
  ]);
}

// Columnas ya en formato de display (PPG, GP, FG%...) se dejan tal cual.
const PRESERVE_LABEL = /^[A-Z0-9%.]{1,6}$/;

function prettifyLabel(col) {
  if (PRESERVE_LABEL.test(col)) return col;
  return col
    .split("_")
    .map((word) => (word.length <= 3 ? word.toUpperCase() : word[0].toUpperCase() + word.slice(1)))
    .join(" ");
}

/** Compara valores de celda: numérico si ambos son number, alfabético
 * (localeCompare es-ES) si no. null/undefined siempre al final. */
function compareCellValues(a, b) {
  const aMissing = a == null;
  const bMissing = b == null;
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1;
  if (bMissing) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), "es", { numeric: true, sensitivity: "base" });
}

/** Tabla con tooltip nativo (title=) por columna del glosario + un
 * <details> con el texto completo debajo. Ordenable por columna (clic en
 * la cabecera: asc / desc / orden original) vía un <button> dentro de
 * cada <th> para foco de teclado accesible. `options.sortable: false` la
 * desactiva para una tabla concreta. */
export function dataTable(records, glossary = {}, options = {}) {
  const { formatters = {}, maxRows = null, hiddenColumns = [], doubleClick = null, rowClass = null, sortable = true } = options;
  // doubleClick acepta un {column, onOpen} o varios (normalizado a Map).
  const doubleClickByColumn = new Map(
    (Array.isArray(doubleClick) ? doubleClick : doubleClick ? [doubleClick] : []).map((d) => [d.column, d.onOpen])
  );
  if (!records || records.length === 0) {
    return emptyState("No hay filas para mostrar.");
  }
  const columns = Object.keys(records[0]).filter((c) => !hiddenColumns.includes(c));
  const baseRows = maxRows ? records.slice(0, maxRows) : records;

  let sortColumn = null;
  let sortDirection = null; // "asc" | "desc" | null = orden original
  const headerCellByColumn = new Map();

  function buildRow(record) {
    return el(
      "tr",
      rowClass && rowClass(record) ? { class: rowClass(record) } : {},
      columns.map((col) => {
        const raw = record[col];
        let formatted = formatters[col] ? formatters[col](raw, record) : raw;
        if (formatted == null) formatted = "—";
        else if (typeof formatted === "number" && !Number.isInteger(formatted)) formatted = formatted.toFixed(2);
        const onOpen = doubleClickByColumn.get(col);
        return el(
          "td",
          onOpen
            ? { class: "cell-clickable", title: "Doble clic para ver el detalle", ondblclick: () => onOpen(record) }
            : {},
          formatted
        );
      })
    );
  }

  const tbody = el("tbody", {});

  function renderBody() {
    const rows =
      sortColumn == null
        ? baseRows
        : [...baseRows].sort((a, b) => {
            const cmp = compareCellValues(a[sortColumn], b[sortColumn]);
            return sortDirection === "desc" ? -cmp : cmp;
          });
    tbody.replaceChildren(...rows.map(buildRow));
  }

  function updateHeaderIndicators() {
    for (const [col, th] of headerCellByColumn) {
      const active = col === sortColumn;
      th.setAttribute("aria-sort", active ? (sortDirection === "desc" ? "descending" : "ascending") : "none");
      const arrow = th.querySelector(".th-sort-arrow");
      if (arrow) arrow.textContent = active ? (sortDirection === "desc" ? "▼" : "▲") : "";
    }
  }

  function headerCell(col) {
    const labelChildren = [
      el("span", { class: "th-label" }, prettifyLabel(col)),
      glossary[col] ? el("span", { class: "info-icon" }, "ⓘ") : null,
    ];
    if (!sortable) {
      return el("th", glossary[col] ? { title: glossary[col] } : {}, labelChildren);
    }
    const th = el("th", { "aria-sort": "none", ...(glossary[col] ? { title: glossary[col] } : {}) }, [
      el(
        "button",
        {
          type: "button",
          class: "th-sort-btn",
          onclick: () => {
            if (sortColumn === col) {
              // asc -> desc -> orden original (deja volver al orden que trajo la tabla).
              sortDirection = sortDirection === "asc" ? "desc" : sortDirection === "desc" ? null : "asc";
              if (sortDirection === null) sortColumn = null;
            } else {
              sortColumn = col;
              sortDirection = "asc";
            }
            updateHeaderIndicators();
            renderBody();
          },
        },
        [...labelChildren, el("span", { class: "th-sort-arrow", "aria-hidden": "true" }, "")]
      ),
    ]);
    headerCellByColumn.set(col, th);
    return th;
  }

  const headerRow = el("tr", {}, columns.map(headerCell));
  renderBody();

  const table = el("table", { class: "data-table" }, [el("thead", {}, headerRow), tbody]);

  const wrap = el("div", { class: "table-wrap" }, table);
  const shadowLeft = el("div", { class: "scroll-shadow scroll-shadow-left" });
  const shadowRight = el("div", { class: "scroll-shadow scroll-shadow-right" });
  const scrollRegion = el("div", { class: "table-scroll-region" }, [wrap, shadowLeft, shadowRight]);
  const container = el("div", {}, [scrollRegion]);

  // Sombra de scroll horizontal (ver .scroll-shadow en components.css).
  // requestAnimationFrame porque scrollWidth recién montado puede no estar
  // listo en el mismo tick. Sin listener de `resize` a propósito -- viviría
  // en `window` y nunca se limpiaría; el de `scroll` muere con `wrap`.
  const updateShadows = () => {
    const maxScroll = wrap.scrollWidth - wrap.clientWidth;
    shadowLeft.classList.toggle("visible", wrap.scrollLeft > 2);
    shadowRight.classList.toggle("visible", maxScroll > 2 && wrap.scrollLeft < maxScroll - 2);
  };
  wrap.addEventListener("scroll", updateShadows, { passive: true });
  requestAnimationFrame(updateShadows);

  const glossaryEntries = Object.entries(glossary).filter(([col]) => columns.includes(col));
  if (glossaryEntries.length > 0) {
    container.append(glossaryExpander(glossaryEntries));
  }
  return container;
}

export function glossaryExpander(entries, title = "Leyenda de estadísticas") {
  const body = el(
    "div",
    { class: "glossary-body" },
    entries.map(([col, text]) => el("p", {}, [el("b", {}, col), ` — ${text}`]))
  );
  return el("details", { class: "glossary" }, [el("summary", {}, title), body]);
}

export function card(children) {
  return el("div", { class: "card" }, children);
}
