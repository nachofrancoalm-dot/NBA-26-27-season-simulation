🌐 [English](README.md) · **Español**

# Notebooks

Narraciones visuales y curadas de tres de las investigaciones del proyecto. Son un complemento a la fuente
de verdad real — los scripts testeados en [`scripts/experiments/`](../scripts/experiments/) y
[`src/backtesting.py`](../src/backtesting.py) — no un sustituto. Cada notebook vuelve a correr el análisis
real sobre `data/processed/` (ya incluido en este repo) y muestra los mismos números que documentan el
propio README del proyecto y `CLAUDE.md`.

- [`01_lineup_synergy_investigation.ipynb`](01_lineup_synergy_investigation.ipynb) — ¿un bonus de "sinergia
  de alineación" ajustado a mano predice el net rating real de una pareja de jugadores? (No — se probaron 5
  efectos candidatos, ninguno sobrevive la validación leave-one-season-out.)
- [`02_contract_year_effect.ipynb`](02_contract_year_effect.ipynb) — ¿los jugadores rinden más en el último
  año de su contrato? (Sin efecto medible en 126 contratos reales, con una regresión de efectos fijos por
  jugador y control de edad.)
- [`03_backtest_calibration_story.ipynb`](03_backtest_calibration_story.ipynb) — cómo un hallazgo de
  "fricción de superequipo" basado en 4 casos resultó ser mayormente un artefacto de calibración al escalar
  el backtesting a 480 temporadas de equipo reales. La lección metodológica más importante del proyecto.

Para volver a correr uno:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_lineup_synergy_investigation.ipynb
```

`02_contract_year_effect.ipynb` necesita `data/raw/contract_data/` (dos CSV de Kaggle, no redistribuidos en
este repo — ver la sección "contract year" del README para saber de dónde sacarlos) para correr desde cero;
los otros dos solo necesitan `data/processed/`, que ya está incluido en el repo.
