import { api } from "../api.js";
import { card, el, emptyState } from "../ui.js";

// Historial en memoria -- se pierde al recargar; explain_question() es de
// un solo turno, así que no hace falta persistirlo.
const history = [];
let rendered = false;

export async function render(container) {
  if (rendered) return; // ya montado -- no perder el historial de chat al volver a la pestaña
  rendered = true;
  container.replaceChildren();
  container.append(el("div", { class: "caption" }, "Cargando…"));

  let status;
  try {
    status = await api.status();
  } catch (err) {
    container.replaceChildren(card([el("h2", {}, "Explicador de resultados en lenguaje natural"), emptyState(err.message)]));
    return;
  }

  if (!status.datasets.explainer) {
    container.replaceChildren(
      card([
        el("h2", {}, "Explicador de resultados en lenguaje natural"),
        emptyState(
          "No se encontró la variable de entorno GROQ_API_KEY. Copia .env.example a .env en la raíz del " +
            "proyecto y rellena tu API key de https://console.groq.com/keys para activar esta pestaña."
        ),
      ])
    );
    return;
  }

  const contextDetails = el("details", { class: "glossary" }, [el("summary", {}, "Ver el contexto (snapshot de datos) que recibe el modelo")]);
  contextDetails.addEventListener(
    "toggle",
    async () => {
      if (!contextDetails.open || contextDetails.dataset.loaded) return;
      contextDetails.dataset.loaded = "1";
      const { snapshot } = await api.explainerContext();
      contextDetails.append(el("pre", { class: "glossary-body", style: "white-space: pre-wrap;" }, snapshot));
    },
    { once: false }
  );

  // Texto de noticias pegado por el usuario -- en memoria como `history`,
  // solo se manda a /explainer/ask si no está vacío.
  const newsDetails = el("details", { class: "glossary" }, [
    el("summary", {}, "Opcional: pegar noticias recientes (lesiones, fichajes, cambios de entrenador...)"),
  ]);
  const newsTextarea = el("textarea", {
    placeholder:
      "Pega aqui texto de articulos o titulares recientes, o usa el buscador de abajo. El modelo " +
      "lo usara como contexto adicional NO verificado -- lo marcara explicitamente como tal en " +
      "sus respuestas, nunca lo mezclara con los datos calculados por el pipeline.",
    rows: "5",
    style: "width: 100%; resize: vertical;",
  });
  const searchQueryInput = el("input", {
    type: "text",
    placeholder: "Ej: lesiones Philadelphia 76ers",
    style: "flex: 1;",
  });
  const searchButton = el("button", { class: "btn" }, "Buscar noticias recientes");
  const searchStatus = el("span", { class: "caption" }, "");
  const searchRow = el("div", { class: "chat-input-row" }, [searchQueryInput, searchButton]);
  newsDetails.append(newsTextarea, searchRow, searchStatus);

  const searchNews = async () => {
    const query = searchQueryInput.value.trim();
    if (!query) return;
    searchButton.disabled = true;
    searchStatus.textContent = "Buscando…";
    try {
      const { news_text } = await api.explainerSearchNews(query);
      newsTextarea.value = news_text || "";
      searchStatus.textContent = news_text ? "" : "Sin resultados para esa búsqueda.";
    } catch (err) {
      searchStatus.textContent = err.message;
    }
    searchButton.disabled = false;
  };
  searchButton.addEventListener("click", searchNews);
  searchQueryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") searchNews();
  });

  const log = el("div", { class: "chat-log", id: "chat-log" });
  const input = el("input", {
    type: "text",
    placeholder: "Ej: ¿por qué Chicago lidera el Este?",
    style: "flex: 1;",
  });
  const sendButton = el("button", { class: "btn" }, "Enviar");
  const form = el("div", { class: "chat-input-row" }, [input, sendButton]);

  container.replaceChildren(
    card([
      el("h2", {}, "Explicador de resultados en lenguaje natural"),
      el(
        "p",
        { class: "caption" },
        "Pregunta lo que quieras sobre los datos ya calculados en las otras pestañas. El modelo (Groq) " +
          "responde SOLO a partir de esos datos -- no inventa cifras ni corre simulaciones nuevas."
      ),
      contextDetails,
      newsDetails,
      log,
      form,
    ])
  );

  renderHistory();

  const send = async () => {
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    input.disabled = true;
    sendButton.disabled = true;
    history.push({ role: "user", content: question });
    renderHistory();
    try {
      const { answer } = await api.explainerAsk(question, newsTextarea.value);
      history.push({ role: "assistant", content: answer });
    } catch (err) {
      history.push({ role: "assistant", content: `Error al consultar el modelo: ${err.message}` });
    }
    renderHistory();
    input.disabled = false;
    sendButton.disabled = false;
    input.focus();
  };

  sendButton.addEventListener("click", send);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") send();
  });
}

function renderHistory() {
  const log = document.getElementById("chat-log");
  if (!log) return;
  log.replaceChildren(
    ...history.map((turn) => el("div", { class: `chat-bubble ${turn.role}` }, turn.content))
  );
  log.scrollTop = log.scrollHeight;
}
