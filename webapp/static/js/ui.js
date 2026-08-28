// ui.js -- helpers de renderizado compartidos entre vistas (tablas, stat
// tiles, glosario, estado vacío, insignia de equipo) -- equivalente a los
// helpers de dashboard/app.py (render_glossary_expander, st.metric, etc.).

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

export function clear(container) {
  container.replaceChildren();
}

export function emptyState(message) {
  return el("div", { class: "empty-state" }, message);
}

export function apiErrorState(err) {
  return emptyState(err.message || "No se encontró el dataset todavía.");
}

/** Placeholder de carga con shimmer, en vez de un texto fijo "Cargando...".
 * `lines` describe el contenido aproximado que va a llegar (una card con
 * título + un par de filas por defecto) para minimizar el salto de layout
 * cuando el dato real reemplaza al esqueleto -- ver progressive-loading en
 * la skill de UI/UX de este proyecto. */
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

/** Selector compacto tipo pill (usado como control LOCAL de "Por partido /
 * Totales" junto a cada tabla, en vez de un toggle global en una barra
 * lateral). `options`: [{value, label}]. Llama a `onChange(value)` al
 * cambiar de selección; el propio botón se marca activo. */
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

/** Insignia genérica: iniciales/texto de respaldo sobre un color de
 * fondo, con intento de imagen real cargada EN VIVO (nunca guardada en
 * el repo -- ver README) que cae de vuelta al texto si la imagen no
 * carga. El texto y la imagen son ALTERNATIVOS, nunca se apilan -- si
 * se dejan los dos en el DOM a la vez, el hueco que deja `object-fit:
 * contain` dentro del círculo deja asomar el texto por un lado (el bug
 * que se vio en Fase 2 como "logos con letras a la izquierda").
 * `teamBadge()` y la foto de jugador del popup son dos casos de este
 * mismo mecanismo. */
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

/** Foto de un jugador -- caso particular de photoBadge() con la URL de
 * headshots de cdn.nba.com. Cae a un círculo con las iniciales del
 * nombre si la foto no carga (jugador sin headshot oficial, red caída
 * en el entorno de pruebas, etc.). */
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

// Columnas ya en formato de display (PPG, GP, FG%, 3PM...) se dejan tal
// cual -- solo se reformatean snake_case/UPPER_SNAKE_CASE crudos de pandas.
const PRESERVE_LABEL = /^[A-Z0-9%.]{1,6}$/;

function prettifyLabel(col) {
  if (PRESERVE_LABEL.test(col)) return col;
  return col
    .split("_")
    .map((word) => (word.length <= 3 ? word.toUpperCase() : word[0].toUpperCase() + word.slice(1)))
    .join(" ");
}

/** Tabla con tooltip nativo (title=) por columna documentada en el glosario
 * + un <details> con el texto completo debajo -- mismo patrón que
 * render_glossary_expander() de dashboard/app.py. */
export function dataTable(records, glossary = {}, options = {}) {
  const { formatters = {}, maxRows = null, hiddenColumns = [], doubleClick = null, rowClass = null } = options;
  // doubleClick acepta un único {column, onOpen} o varios a la vez
  // (p.ej. Premios individuales: doble clic en el jugador Y en el
  // equipo de la misma tabla) -- se normaliza a un Map columna -> onOpen.
  const doubleClickByColumn = new Map(
    (Array.isArray(doubleClick) ? doubleClick : doubleClick ? [doubleClick] : []).map((d) => [d.column, d.onOpen])
  );
  if (!records || records.length === 0) {
    return emptyState("No hay filas para mostrar.");
  }
  const columns = Object.keys(records[0]).filter((c) => !hiddenColumns.includes(c));
  const rows = maxRows ? records.slice(0, maxRows) : records;

  const headerRow = el(
    "tr",
    {},
    columns.map((col) =>
      el("th", glossary[col] ? { title: glossary[col] } : {}, [
        prettifyLabel(col),
        glossary[col] ? el("span", { class: "info-icon" }, "ⓘ") : null,
      ])
    )
  );

  const bodyRows = rows.map((record) =>
    el(
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
    )
  );

  const table = el("table", { class: "data-table" }, [el("thead", {}, headerRow), el("tbody", {}, bodyRows)]);

  const wrap = el("div", { class: "table-wrap" }, table);
  const shadowLeft = el("div", { class: "scroll-shadow scroll-shadow-left" });
  const shadowRight = el("div", { class: "scroll-shadow scroll-shadow-right" });
  const scrollRegion = el("div", { class: "table-scroll-region" }, [wrap, shadowLeft, shadowRight]);
  const container = el("div", {}, [scrollRegion]);

  // Sombra de scroll horizontal -- ver el comentario de .scroll-shadow en
  // components.css. Se recalcula al hacer scroll; requestAnimationFrame
  // porque scrollWidth recién montado puede no estar listo en el mismo
  // tick que el append. Sin listener de `resize` a propósito -- varias
  // vistas recrean su tabla en cada clic (re-simular bracket/calendario),
  // y acumular listeners de `window` que nunca se limpian sí sería un
  // leak real; el de `scroll` vive y muere con `wrap`, que sí se
  // descarta entero al re-renderizar.
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
