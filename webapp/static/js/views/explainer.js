import { api } from "../api.js";
import { card, el, emptyState } from "../ui.js";

// Historial en memoria del módulo -- se pierde al recargar la página,
// igual que st.session_state["explainer_history"] se pierde al
// reiniciar el server de Streamlit. explain_question() es de un solo
// turno (no manda historial al modelo), así que no hace falta más.
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
      const { answer } = await api.explainerAsk(question);
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
