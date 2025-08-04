from zenml.client import Client
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os

# ------------------------------------------------------------------ #
# 1.  Grab the most recent run of the pipeline and load artefacts
# ------------------------------------------------------------------ #
def load_latest_run_outputs(pipeline_name: str):
    """
    Return a dict  {step_name -> {output_name -> python_object}}
    Compatible with zenml 0.84.0 (outputs are lists).
    """
    client = Client()

    # Most-recent run comes first
    runs = client.list_pipeline_runs(pipeline_name=pipeline_name, size=1)
    if not runs:
        raise ValueError(f"No runs found for pipeline '{pipeline_name}'")

    run = client.get_pipeline_run(runs[0].id)
    artefacts = {}

    for step_name, step_view in run.steps.items():
        artefacts[step_name] = {}
        for output_name, output_list in step_view.outputs.items():
            if not output_list:
                continue                       # empty output
            artefacts[step_name][output_name] = output_list[0].load()

    return artefacts, run


# ------------------------------------------------------------------ #
# 2.  Pull the objects we need (model, train/test, predictions)
# ------------------------------------------------------------------ #
def extract_objects(artefacts, book_name: str | None = None):
    """Return train_series, test_series and the fitted SARIMA model."""
    # modelling_data DataFrame (created in prepare_modelling_data_step)
    for d in artefacts.values():
        for name, obj in d.items():
            if "modelling_data" in name and isinstance(obj, pd.DataFrame):
                modelling_df = obj
                break
        else:
            continue
        break
    else:
        raise RuntimeError("Could not locate 'modelling_data' artefact")

    # choose book
    if not book_name:
        book_name = modelling_df["book_name"].unique()[0]

    book_df = modelling_df[modelling_df.book_name == book_name]
    train = book_df[book_df.data_type == "train"].set_index("date")["volume"].sort_index()
    test  = book_df[book_df.data_type == "test"].set_index("date")["volume"].sort_index()

    # SARIMA model (output 'trained_model' of train_arima_optuna_step)
    for d in artefacts.values():
        for name, obj in d.items():
            if "trained_model" in name:
                model = obj
                break
        else:
            continue
        break
    else:
        raise RuntimeError("Could not find a trained model artefact")

    return train, test, model, book_name


# ------------------------------------------------------------------ #
# 3.  Simple plot that uses the stored test-set predictions
# ------------------------------------------------------------------ #

def plot_test_predictions(artefacts, title="SARIMA test-set performance", save_path="plots"):
    # load test_predictions DataFrame
    for d in artefacts.values():
        for name, obj in d.items():
            if "test_predictions" in name and isinstance(obj, pd.DataFrame):
                preds = obj
                break
        else:
            continue
        break
    else:
        raise RuntimeError("No 'test_predictions' artefact found")

    # Calculate metrics
    mae  = preds.absolute_error.mean()
    mape = (preds.absolute_error / preds.actual).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds.date, y=preds.actual,
                             mode="lines", name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=preds.date, y=preds.predicted,
                             mode="lines", name="Predicted", line=dict(color="red")))

    # Add MAE/MAPE as text annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"MAE: {mae:,.2f}<br>MAPE: {mape:,.2f}%",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Volume",
                      template="plotly_white", width=900, height=500)

    print(f"MAE:  {mae:,.2f}")
    print(f"MAPE: {mape:,.2f}%")

    # Save the plot
    filename = "test_predictions.html"
    filepath = os.path.join(save_path, filename)
    fig.write_html(filepath)
    print(f"Plot saved to: {filepath}")

    # fig.show()  # Commented out to prevent massive HTML output
    return fig

# ------------------------------------------------------------------ #
# 5.  CLI entry-point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    PIPELINE_NAME = "book_sales_arima_pipeline"        # <- your pipeline
    BOOK          = None                               # or e.g. "The Alchemist"
    PLOTS_FOLDER  = "plots"                            # plots folder

    # Ensure plots folder exists
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    artefacts, run = load_latest_run_outputs(PIPELINE_NAME)
    print(f"Loaded artefacts from run: {run.name}")

    # --------  Option A: use stored test-set predictions  -------- #
    plot_test_predictions(artefacts,
        title="SARIMA – test-set predictions (ZenML 0.84.0)",
        save_path=PLOTS_FOLDER)
