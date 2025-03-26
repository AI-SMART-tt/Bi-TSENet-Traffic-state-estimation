import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime
import logging
import sys
warnings.filterwarnings('ignore')

def setup_logger(log_dir='./logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/traffic_model_analysis_{timestamp}.log"
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file, 'w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return log_file

def load_model_predictions(base_dir='./output'):
    logging.info("Loading data...")
    
    models = ['Bi-TSENet', 'SVR', 'LSTM', 'TCN', 'Con-LSTM', 'STGCN', 'ASTGCN', 'GMAN', 'Graph-WaveNet']
    
    predictions = {}
    true_values = None
    available_models = []
    
    for model in models:
        pred_path = f'{base_dir}/{model}/values/predictions.csv'
        
        if not os.path.exists(pred_path):
            logging.warning(f"Warning: Prediction file for {model} not found ({pred_path})")
            continue
        
        try:
            predictions[model] = pd.read_csv(pred_path)
            
            predictions[model]['Time'] = pd.to_datetime(predictions[model]['Time'])
            predictions[model].set_index('Time', inplace=True)
            
            available_models.append(model)
            
            if true_values is None:
                true_path = f'{base_dir}/{model}/values/actual_values.csv'
                if os.path.exists(true_path):
                    true_values = pd.read_csv(true_path)
                    true_values['Time'] = pd.to_datetime(true_values['Time'])
                    true_values.set_index('Time', inplace=True)
        except Exception as e:
            logging.error(f"Error loading {model} data: {str(e)}")
    
    if true_values is None:
        true_paths = glob.glob(f'{base_dir}/*/values/actual_values.csv')
        if true_paths:
            true_values = pd.read_csv(true_paths[0])
            true_values['Time'] = pd.to_datetime(true_values['Time'])
            true_values.set_index('Time', inplace=True)
        else:
            raise FileNotFoundError("Unable to find ground truth file. Ensure at least one model directory contains actual_values.csv")
    
    if len(available_models) == 0:
        raise ValueError("Failed to load any model data. Please check data paths and formats.")
    
    logging.info(f"Successfully loaded predictions for {len(predictions)} models")
    logging.info(f"Available models: {', '.join(available_models)}")
    
    return predictions, true_values, available_models

def align_data(predictions, true_values, models):
    logging.info("Aligning data...")
    
    all_dfs = [true_values] + [predictions[model] for model in models]
    common_start = max(df.index.min() for df in all_dfs)
    common_end = min(df.index.max() for df in all_dfs)
    
    logging.info(f"Common time range: {common_start} to {common_end}")
    
    time_diff = true_values.index[1] - true_values.index[0]
    time_range = pd.date_range(start=common_start, end=common_end, freq=time_diff)
    
    aligned_true = true_values.reindex(time_range)
    aligned_predictions = {}
    
    for model in models:
        aligned_predictions[model] = predictions[model].reindex(time_range)
    
    aligned_true = aligned_true.ffill(limit=2).bfill(limit=1)
    for model in models:
        aligned_predictions[model] = aligned_predictions[model].ffill(limit=2).bfill(limit=1)
    
    valid_idx = ~aligned_true.isna().any(axis=1)
    for model in models:
        valid_idx = valid_idx & ~aligned_predictions[model].isna().any(axis=1)
    
    aligned_true = aligned_true[valid_idx]
    for model in models:
        aligned_predictions[model] = aligned_predictions[model][valid_idx]
    
    valid_data_percent = (len(aligned_true) / len(time_range)) * 100
    logging.info(f"Data points after alignment: {len(aligned_true)} ({valid_data_percent:.2f}% of original time range)")
    
    return aligned_predictions, aligned_true

def calculate_metrics(predictions, true_values, nodes):
    metrics = {'MAE': {}, 'RMSE': {}, 'MAPE': {}, 'R2': {}}
    
    for model, pred_df in predictions.items():
        metrics['MAE'][model] = {}
        metrics['RMSE'][model] = {}
        metrics['MAPE'][model] = {}
        metrics['R2'][model] = {}
        
        for node in nodes:
            y_true = true_values[node].values
            y_pred = pred_df[node].values
            
            metrics['MAE'][model][node] = np.mean(np.abs(y_pred - y_true))
            
            metrics['RMSE'][model][node] = np.sqrt(np.mean((y_pred - y_true)**2))
            
            mask = y_true != 0
            if np.sum(mask) > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                metrics['MAPE'][model][node] = mape
            else:
                metrics['MAPE'][model][node] = np.nan
            
            if np.var(y_true) > 0:
                ss_total = np.sum((y_true - np.mean(y_true))**2)
                ss_residual = np.sum((y_true - y_pred)**2)
                metrics['R2'][model][node] = 1 - (ss_residual / ss_total)
            else:
                metrics['R2'][model][node] = np.nan
    
    return metrics

def perform_statistical_tests(predictions, true_values, models, our_model='Bi-TSENet', test_method='paired_t_test'):
    logging.info(f"Performing statistical tests (method: {test_method}, using R2 metric)...")
    
    nodes = [col for col in true_values.columns if 'Node_' in col]
    if not nodes:
        nodes = true_values.columns.tolist()
    
    metrics = calculate_metrics(predictions, true_values, nodes)
    
    test_results = {}
    
    if our_model not in models:
        logging.warning(f"Warning: Our model {our_model} is not in the available model list, cannot perform comparison")
        return {}, metrics
    
    for baseline_model in models:
        if baseline_model == our_model:
            continue
        
        all_our_model_r2 = []
        all_baseline_r2 = []
        improvement_percentages = []
        
        node_p_values = []
        
        for node in nodes:
            y_true = true_values[node].values
            y_pred_our = predictions[our_model][node].values
            y_pred_baseline = predictions[baseline_model][node].values
            
            if np.var(y_true) > 0:
                ss_total = np.sum((y_true - np.mean(y_true))**2)
                ss_residual_our = np.sum((y_true - y_pred_our)**2)
                r2_our = 1 - (ss_residual_our / ss_total)
                
                ss_residual_baseline = np.sum((y_true - y_pred_baseline)**2)
                r2_baseline = 1 - (ss_residual_baseline / ss_total)
                
                all_our_model_r2.append(r2_our)
                all_baseline_r2.append(r2_baseline)
                
                if r2_baseline > 0:
                    node_improvement = ((r2_our - r2_baseline) / abs(r2_baseline)) * 100
                else:
                    node_improvement = 100 if r2_our > r2_baseline else 0
                    
                improvement_percentages.append(node_improvement)
                
                try:
                    if test_method == 'paired_t_test':
                        sq_error_baseline = (y_true - y_pred_baseline)**2
                        sq_error_our = (y_true - y_pred_our)**2
                        t_stat, p_value = stats.ttest_rel(sq_error_baseline, sq_error_our)
                        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
                        
                    elif test_method == 'sign_test':
                        diff = (y_true - y_pred_baseline)**2 - (y_true - y_pred_our)**2
                        n_positive = np.sum(diff > 0)
                        n_total = len(diff)
                        p_value = 1 - stats.binom.cdf(n_positive-1, n_total, 0.5)
                        
                    elif test_method == 'bootstrap':
                        n_bootstrap = 1000
                        bootstrap_diffs = []
                        
                        for _ in range(n_bootstrap):
                            idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
                            y_true_boot = y_true[idx]
                            y_pred_our_boot = y_pred_our[idx]
                            y_pred_baseline_boot = y_pred_baseline[idx]
                            
                            if np.var(y_true_boot) > 0:
                                ss_total_boot = np.sum((y_true_boot - np.mean(y_true_boot))**2)
                                
                                ss_residual_our_boot = np.sum((y_true_boot - y_pred_our_boot)**2)
                                r2_our_boot = 1 - (ss_residual_our_boot / ss_total_boot)
                                
                                ss_residual_baseline_boot = np.sum((y_true_boot - y_pred_baseline_boot)**2)
                                r2_baseline_boot = 1 - (ss_residual_baseline_boot / ss_total_boot)
                                
                                bootstrap_diffs.append(r2_our_boot - r2_baseline_boot)
                        
                        p_value = np.mean(np.array(bootstrap_diffs) <= 0)
                    
                    else:
                        sq_error_baseline = (y_true - y_pred_baseline)**2
                        sq_error_our = (y_true - y_pred_our)**2
                        t_stat, p_value = stats.ttest_rel(sq_error_baseline, sq_error_our)
                        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
                    
                    node_p_values.append(p_value)
                    
                except Exception as e:
                    logging.error(f"Error performing {test_method} test for node {node}: {str(e)}")
        
        mean_r2_our = np.mean(all_our_model_r2) if all_our_model_r2 else 0
        mean_r2_baseline = np.mean(all_baseline_r2) if all_baseline_r2 else 0
        
        if mean_r2_baseline > 0:
            overall_improvement = ((mean_r2_our - mean_r2_baseline) / abs(mean_r2_baseline)) * 100
        else:
            overall_improvement = 100 if mean_r2_our > mean_r2_baseline else 0
        
        if node_p_values:
            node_p_values = [max(p, 1e-10) for p in node_p_values]
            combined_stat = -2 * np.sum(np.log(node_p_values))
            combined_p = stats.chi2.sf(combined_stat, df=2*len(node_p_values))
            
            test_results[f'{our_model} vs. {baseline_model}'] = {
                'p-value': combined_p,
                'test-method': test_method,
                'significant': combined_p < 0.05,
                'baseline_r2': mean_r2_baseline,
                'our_model_r2': mean_r2_our,
                'improvement': overall_improvement,
                'sample_size': len(node_p_values),
                'node_improvements': improvement_percentages
            }
        else:
            logging.warning(f"Warning: Comparison between {our_model} vs. {baseline_model} did not generate valid p-values")
    
    return test_results, metrics

def summarize_metrics(metrics, models, nodes):
    summary = {}
    
    for metric_name in metrics:
        summary[metric_name] = {}
        
        for model in models:
            node_values = []
            for node in nodes:
                if node in metrics[metric_name][model]:
                    node_values.append(metrics[metric_name][model][node])
            
            if node_values:
                mean_value = np.mean(node_values)
                std_value = np.std(node_values)
                summary[metric_name][model] = {
                    'mean': mean_value,
                    'std': std_value
                }
    
    return summary

def print_test_results(test_results, test_method):
    logging.info(f"\n====== Model Performance Comparison Results ({test_method} test based on R2 metric) ======")
    
    headers = ["Comparison", "p-value", "Significant", "Sample Size", "Baseline R2", "Bi-TSENet R2", "Improvement %"]
    rows = []
    
    for comparison, result in test_results.items():
        significant = 'Yes *' if result['significant'] else 'No'
        
        if result['p-value'] < 0.0001:
            p_value_str = f"{result['p-value']:.4e}"
        else:
            p_value_str = f"{result['p-value']:.4f}"
            
        rows.append([
            comparison, 
            p_value_str,
            significant,
            result['sample_size'],
            f"{result['baseline_r2']:.4f}",
            f"{result['our_model_r2']:.4f}",
            f"{result['improvement']:.2f}%"
        ])
    
    rows.sort(key=lambda x: float(x[6].replace('%', '')), reverse=True)
    
    table_str = tabulate(rows, headers, tablefmt="grid")
    logging.info("\n" + table_str)
    
    logging.info("\nNote: '*' indicates statistically significant difference at α=0.05 level")
    logging.info(f"      Test method: {test_method}")
    logging.info("      Lower p-values indicate stronger evidence that Bi-TSENet outperforms the baseline model")
    logging.info("      Improvement % = Percentage increase in R2 of Bi-TSENet relative to baseline model")
    logging.info("      R2 closer to 1 indicates better model performance")

def print_summary_metrics(summary, models, our_model='Bi-TSENet'):
    logging.info("\n====== Model Performance Metrics Summary ======")
    
    for metric_name in summary:
        logging.info(f"\n{metric_name} Metric Summary:")
        
        headers = ["Model", "Mean", "Std Dev"]
        rows = []
        
        for model in models:
            if model in summary[metric_name]:
                model_name = f"{model} *" if model == our_model else model
                rows.append([
                    model_name,
                    f"{summary[metric_name][model]['mean']:.4f}",
                    f"{summary[metric_name][model]['std']:.4f}"
                ])
        
        reverse_sort = metric_name == 'R2'
        rows.sort(key=lambda x: float(x[1]), reverse=reverse_sort)
        
        table_str = tabulate(rows, headers, tablefmt="grid")
        logging.info("\n" + table_str)
        
        if metric_name in ['MAE', 'RMSE', 'MAPE']:
            logging.info("Note: Lower values indicate better performance")
        elif metric_name == 'R2':
            logging.info("Note: Higher values indicate better performance")
        logging.info("      * marks our proposed model")

def export_to_csv(test_results, results_dir):
    test_df = pd.DataFrame([
        {
            'Comparison': comp,
            'p-value': res['p-value'],
            'Significant': 'Yes' if res['significant'] else 'No',
            'Baseline_R2': res['baseline_r2'],
            'Our_Model_R2': res['our_model_r2'],
            'Improvement_Percent': res['improvement'],
            'Sample_Size': res['sample_size']
        } for comp, res in test_results.items()
    ])
    
    test_df = test_df.sort_values(by='Improvement_Percent', ascending=False)
    test_df.to_csv(f"{results_dir}/statistical_test_results.csv", index=False)
    
    logging.info(f"Statistical test results exported to CSV in {results_dir}")

def save_results(test_results, metrics_summary, models, test_method, our_model='Bi-TSENet', output_dir='./results'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{output_dir}/{timestamp}_{test_method}"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    export_to_csv(test_results, results_dir)
    
    try:
        plt.switch_backend('Agg')
        plt.style.use('seaborn-darkgrid')
        
        for metric_name in metrics_summary:
            plt.figure(figsize=(14, 8))
            
            model_names = []
            values = []
            errors = []
            colors = []
            
            for model in models:
                if model in metrics_summary[metric_name]:
                    model_names.append(model)
                    values.append(metrics_summary[metric_name][model]['mean'])
                    errors.append(metrics_summary[metric_name][model]['std'])
                    colors.append('#E63946' if model == our_model else '#457B9D')
            
            if metric_name in ['MAE', 'RMSE', 'MAPE']:
                sorted_indices = np.argsort(values)
            else:
                sorted_indices = np.argsort(values)[::-1]
            
            sorted_models = [model_names[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            sorted_errors = [errors[i] for i in sorted_indices]
            sorted_colors = [colors[i] for i in sorted_indices]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            bars = ax.bar(sorted_models, sorted_values, yerr=sorted_errors, capsize=6, 
                        color=sorted_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
            
            for bar, val in zip(bars, sorted_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', rotation=0, 
                        size=11, fontweight='bold')
            
            if metric_name == 'R2':
                min_val = min(values) - 0.02
                min_val = max(0, min_val - 0.05)
                if min_val > 0.7:
                    ax.set_ylim(min_val, 1.02)
                ax.set_ylabel(f'{metric_name} Score', fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel(f'{metric_name} Value', fontsize=14, fontweight='bold')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f'Model Performance Comparison - {metric_name}', fontsize=16, fontweight='bold')
            plt.xlabel('Models', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#E63946', edgecolor='black', label='Bi-TSENet (Our Model)', linewidth=1.5),
                Patch(facecolor='#457B9D', edgecolor='black', label='Baseline Models', linewidth=1.5)
            ]
            plt.legend(handles=legend_elements, loc='best', fontsize=12)
            
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{metric_name}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        if test_results:
            plt.figure(figsize=(14, 8))
            
            comparisons = []
            improvements = []
            significances = []
            
            for comparison, result in test_results.items():
                comp_label = comparison.replace('Bi-TSENet vs. ', '')
                comparisons.append(comp_label)
                improvements.append(result['improvement'])
                significances.append(result['significant'])
            
            sorted_indices = np.argsort(improvements)[::-1]
            sorted_comparisons = [comparisons[i] for i in sorted_indices]
            sorted_improvements = [improvements[i] for i in sorted_indices]
            sorted_significances = [significances[i] for i in sorted_indices]
            
            colors = ['#2A9D8F' if sig else '#81B29A' for sig in sorted_significances]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            bars = ax.bar(sorted_comparisons, sorted_improvements, color=colors, 
                        edgecolor='black', linewidth=1.5, alpha=0.8)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                        height + 0.5 if height > 0 else height - 2.0,
                        f"{sorted_improvements[i]:.2f}%", 
                        ha='center', va='bottom' if height > 0 else 'top', 
                        rotation=0, size=12, fontweight='bold')
            
            ax.axhline(y=0, color='#E63946', linestyle='-', linewidth=2, alpha=0.7)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.title(f'Bi-TSENet Performance Improvement vs. Baseline Models\n({test_method} test)', 
                    fontsize=16, fontweight='bold')
            plt.ylabel('R2 Improvement Percentage (%)', fontsize=14, fontweight='bold')
            plt.xlabel('Baseline Models', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12)
            
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2A9D8F', edgecolor='black', label='Statistically Significant (p<0.05)', linewidth=1.5),
                Patch(facecolor='#81B29A', edgecolor='black', label='Not Statistically Significant', linewidth=1.5)
            ]
            plt.legend(handles=legend_elements, loc='best', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/improvements_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            if 'node_improvements' in list(test_results.values())[0]:
                plt.figure(figsize=(14, 8))
                
                node_improvements_data = []
                labels = []
                
                for comparison, result in test_results.items():
                    comp_label = comparison.replace('Bi-TSENet vs. ', '')
                    labels.append(comp_label)
                    node_improvements_data.append(result['node_improvements'])
                
                sorted_indices = np.argsort([np.mean(imp) for imp in node_improvements_data])[::-1]
                sorted_data = [node_improvements_data[i] for i in sorted_indices]
                sorted_labels = [labels[i] for i in sorted_indices]
                
                fig, ax = plt.subplots(figsize=(14, 8))
                box = ax.boxplot(sorted_data, patch_artist=True, labels=sorted_labels,
                               whiskerprops={'linewidth': 2},
                               boxprops={'linewidth': 2, 'color': 'black'},
                               medianprops={'linewidth': 2, 'color': '#E63946'},
                               flierprops={'marker': 'o', 'markerfacecolor': '#F1FAEE', 'markeredgecolor': 'black'})
                
                for patch in box['boxes']:
                    patch.set_facecolor('#457B9D')
                    patch.set_alpha(0.8)
                
                ax.axhline(y=0, color='#E63946', linestyle='--', linewidth=2.5, alpha=0.7)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.title('Distribution of Node-level Improvement Percentages', 
                        fontsize=16, fontweight='bold')
                plt.ylabel('R2 Improvement Percentage (%)', fontsize=14, fontweight='bold')
                plt.xlabel('Baseline Models', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
                plt.yticks(fontsize=12)
                
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                
                plt.tight_layout()
                plt.savefig(f"{results_dir}/node_improvements_boxplot.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        logging.info(f"\nEnhanced visualizations saved to: {results_dir}")
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
    
    logging.info(f"\nComplete analysis results saved to: {results_dir}")
    return results_dir

def main():
    try:
        base_dir = './output'
        output_dir = './results'
        our_model = 'Bi-TSENet'
        
        # Options: 'paired_t_test', 'sign_test', 'bootstrap'
        test_method = 'bootstrap'
        
        log_file = setup_logger()
        
        logging.info("=" * 80)
        logging.info("Traffic Flow Prediction Model Comparison Analysis Tool")
        logging.info("=" * 80)
        logging.info(f"Using {test_method} test method with R2 metric for performance comparison")
        
        predictions, true_values, available_models = load_model_predictions(base_dir)
        
        aligned_predictions, aligned_true = align_data(predictions, true_values, available_models)
        
        nodes = [col for col in aligned_true.columns if 'Node_' in col]
        if not nodes:
            nodes = aligned_true.columns.tolist()
        
        logging.info(f"\nNumber of traffic nodes analyzed: {len(nodes)}")
        logging.info(f"Node list: {', '.join(nodes)}")
        
        if our_model not in available_models:
            logging.error(f"Error: Our model {our_model} is not in the available model list, cannot perform performance comparison")
            return
        
        test_results, metrics = perform_statistical_tests(
            aligned_predictions, 
            aligned_true, 
            available_models, 
            our_model,
            test_method
        )
        
        metrics_summary = summarize_metrics(metrics, available_models, nodes)
        
        print_test_results(test_results, test_method)
        print_summary_metrics(metrics_summary, available_models, our_model)
        
        results_dir = save_results(test_results, metrics_summary, available_models, test_method, our_model, output_dir)
        
        logging.info("\nDetailed node-level R2 values:")
        for model in available_models:
            logging.info(f"\n{model} R2 values by node:")
            for node in nodes:
                if node in metrics['R2'][model]:
                    logging.info(f"  {node}: {metrics['R2'][model][node]:.4f}")
        
        logging.info("\nAnalysis complete!")
        logging.info(f"Detailed results saved to: {results_dir}")
        logging.info(f"Log file saved to: {log_file}")
        
    except Exception as e:
        logging.error(f"\nError during analysis: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()