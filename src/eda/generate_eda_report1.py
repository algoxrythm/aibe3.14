import os
import csv
import argparse
from datetime import datetime

import pandas as pd
import sweetviz as sv
from ydata_profiling import ProfileReport
import plotly.express as px

from rich.console import Console
from rich import print
from rich.table import Table

console = Console()

# -------------------- Config --------------------
DEFAULT_SAMPLE_SIZE = 500
MISSING_THRESHOLD = 0.3
REPORTS_DIR = "reports"

# -------------------- Auto Delimiter --------------------
def detect_delimiter(file_path, sample_size=2048):
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(sample_size)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return ','  # fallback

# -------------------- Data Load --------------------
def load_dataset(file_path):
    try:
        delimiter = detect_delimiter(file_path)
        df = pd.read_csv(file_path, sep=delimiter)
        console.log(f"✅ Loaded: [green]{file_path}[/] with delimiter '[cyan]{delimiter}[/]'")
        return df
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load:[/] {file_path}\n{e}")
        return None

# -------------------- Output Directory --------------------
def create_output_dir(dataset_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"{REPORTS_DIR}/{dataset_name}_{timestamp}"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "viz"), exist_ok=True)
    return output_path

# -------------------- Smart Conversion --------------------
def smart_convert_columns(df):
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].str.match(r"\d{4}-\d{2}-\d{2}").sum() > 0.7 * len(df):
                df[col] = pd.to_datetime(df[col], errors='ignore')
            elif df[col].str.match(r"^\d+$").sum() > 0.9 * len(df):
                df[col] = df[col].astype(str)
    return df

# -------------------- Column Type Summary --------------------
def summarize_column_types(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    text_cols = [col for col in cat_cols if df[col].str.len().mean() > 40]
    return num_cols, cat_cols, text_cols, date_cols

def print_column_type_table(num, cat, text, date):
    table = Table(title="📋 Column Type Summary")
    table.add_column("Type", style="cyan bold")
    table.add_column("Columns", style="magenta")

    table.add_row("Numerical", ", ".join(num) or "—")
    table.add_row("Categorical", ", ".join(cat) or "—")
    table.add_row("Text-like", ", ".join(text) or "—")
    table.add_row("Date/Time", ", ".join(date) or "—")

    console.print(table)

# -------------------- Missing Values --------------------
def check_missing_values(df):
    missing = df.isnull().mean().sort_values(ascending=False)
    high_missing = missing[missing > MISSING_THRESHOLD]
    if not high_missing.empty:
        console.print(f"[red]⚠️ High missing value columns (> {MISSING_THRESHOLD:.0%}):[/]")
        for col, pct in high_missing.items():
            console.print(f"   • [bold]{col}[/]: {pct:.1%}")

# -------------------- Sample Export --------------------
def save_samples(df, output_path, dataset_name):
    sample_path = f"{output_path}/{dataset_name}_sample.csv"
    df.sample(n=min(DEFAULT_SAMPLE_SIZE, len(df))).to_csv(sample_path, index=False)
    console.log(f"🧪 Sample saved: [green]{sample_path}[/]")

# -------------------- Reports --------------------
def generate_profile_report(df, output_path, dataset_name):
    console.log("📊 Generating [cyan]pandas profiling[/] report...")
    profile = ProfileReport(df, title=f"{dataset_name} Profiling Report", explorative=True)
    profile.to_file(f"{output_path}/{dataset_name}_profiling.html")
    console.log(f"✅ Profiling saved")

def generate_sweetviz_report(df, output_path, dataset_name):
    console.log("📈 Generating [magenta]Sweetviz[/] report...")
    report = sv.analyze(df)
    report.show_html(f"{output_path}/{dataset_name}_sweetviz.html")
    console.log(f"✅ Sweetviz saved")

# -------------------- Visuals --------------------
def generate_correlation_heatmap(df, output_path, dataset_name):
    num_df = df.select_dtypes(include=['int64', 'float64'])
    if num_df.empty:
        console.print(f"[yellow]⚠️ No numerical columns to plot correlation heatmap.[/]")
        return
    fig = px.imshow(num_df.corr(), text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu", title="Correlation Heatmap")
    fig.write_html(f"{output_path}/viz/{dataset_name}_correlation_heatmap.html")
    console.log(f"🎨 Correlation heatmap saved")

def plot_categorical_distributions(df, cat_cols, output_path, dataset_name):
    for col in cat_cols:
        if df[col].nunique() <= 20:
            fig = px.bar(df[col].value_counts().reset_index(),
                         x='index', y=col,
                         labels={'index': col, col: "Count"},
                         title=f"Distribution of {col}")
            fig.update_traces(textposition='outside')
            fig.write_html(f"{output_path}/viz/{dataset_name}_{col}_bar.html")
    console.log(f"📊 Categorical bar charts saved")

# -------------------- Run EDA --------------------
def run_eda(file_path, args):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    df = load_dataset(file_path)
    if df is None:
        return

    df = smart_convert_columns(df)
    output_path = create_output_dir(dataset_name)

    num, cat, text, date = summarize_column_types(df)
    print_column_type_table(num, cat, text, date)
    check_missing_values(df)

    if not args.skip_profile:
        generate_profile_report(df, output_path, dataset_name)

    if not args.skip_sweetviz:
        generate_sweetviz_report(df, output_path, dataset_name)

    if not args.skip_sample:
        save_samples(df, output_path, dataset_name)

    generate_correlation_heatmap(df, output_path, dataset_name)
    plot_categorical_distributions(df, cat, output_path, dataset_name)

    console.print(f"\n[bold green]✅ All EDA complete for:[/] {dataset_name}\n")

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔥 Visual + Modular EDA Script")
    parser.add_argument("--input", type=str, help="Path to single CSV file", required=False)
    parser.add_argument("--all", action="store_true", help="Run on all CSVs in data/raw/")
    parser.add_argument("--skip-profile", action="store_true", help="Skip Pandas Profiling")
    parser.add_argument("--skip-sweetviz", action="store_true", help="Skip Sweetviz")
    parser.add_argument("--skip-sample", action="store_true", help="Skip Sample CSV Export")

    args = parser.parse_args()

    if args.all:
        files = [f for f in os.listdir("data/raw") if f.endswith(".csv")]
        for f in files:
            run_eda(os.path.join("data/raw", f), args)
    elif args.input:
        run_eda(args.input, args)
    else:
        console.print("[red]❌ Please provide either --input or --all[/]")

