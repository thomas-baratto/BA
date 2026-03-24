import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Convert summary CSV to a nicely formatted LaTeX table")
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, help='Output LaTeX file (default: stdout)', default=None)
    parser.add_argument('--caption', type=str, help='Table caption', default="Model Performance Summary")
    parser.add_argument('--label', type=str, help='Table label', default="tab:results")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error reading {args.input_csv}: {e}", file=sys.stderr)
        sys.exit(1)

    # Drop Folder column as it's typically too long to fit in a standard table
    if 'Folder' in df.columns:
        df = df.drop('Folder', axis=1)

    # Convert numeric columns to nicely formatted strings
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col in ['R2', 'MAPE', 'RMSE', 'MAE']:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if abs(x) < 1000 else f"{x:.1f}")
            elif col in ['Time(s)']:
                df[col] = df[col].apply(lambda x: f"{x:.2f}")
            elif col in ['MSE']:
                df[col] = df[col].apply(lambda x: f"{x:.2e}")
            else:
                # Default numeric formatting
                df[col] = df[col].apply(lambda x: f"{x:.4g}")

    # Standardize column names (optional, e.g., make them Title Case or replace _ with space)
    df.columns = [col.replace('_', ' ').replace('%', '\\%') for col in df.columns]

    # Generate the LaTeX string using booktabs for a professional look
    # Note: older pandas might handle 'booktabs' differently, but it generally works.
    try:
        latex_str = df.to_latex(
            index=False, 
            caption=args.caption, 
            label=args.label,
            column_format="l" * df.shape[1], # Default alignment
            escape=False # Since we added LaTeX escapes like \\%
        )
        
        # Add booktabs commands manually if not supported correctly by all pandas versions, 
        # but to_latex usually has `booktabs=True` in most recent versions.
        # Let's use `booktabs=True` explicitly.
    except TypeError:
         # Fallback for pandas versions that don't support some arguments on to_latex
         pass
         
    # To be safe across pandas versions (since pandas 2.0 has moved to_latex to Styler.to_latex):
    if hasattr(df.style, 'to_latex'):
        latex_str = df.style.hide(axis='index') \
            .format(precision=4) \
            .to_latex(
                caption=args.caption,
                label=args.label,
                hrules=True # uses booktabs
            )
    else:
        # Fallback for older pandas
        latex_str = df.to_latex(index=False, escape=False)

    # If booktabs was used in Styler, replace standard lines with booktabs lines for better formatting
    latex_str = latex_str.replace('\\toprule', '\\toprule\n').replace('\\midrule', '\\midrule\n').replace('\\bottomrule', '\\bottomrule\n')

    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table successfully saved to {args.output}")
    else:
        # Print directly to stdout
        print(latex_str)

if __name__ == "__main__":
    main()
