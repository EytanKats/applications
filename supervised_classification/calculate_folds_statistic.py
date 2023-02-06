import os
import pandas as pd


from settings import settings
from simple_converge.utils.dataset_utils import calculate_folds_statistic, plot_folds_metrics


def calculate_statistics(file_name, params):

    # Get metrics data files
    output_folder = os.path.join(params['output_folder'], 'test')
    metrics_dfs = [pd.read_csv(os.path.join(output_folder, str(fold), file_name), index_col=0)
                   for fold in params['active_folds']]

    # Plot
    plot_folds_metrics(metrics_dfs, save_plot=True, output_dir=output_folder)

    statistics_df = calculate_folds_statistic(metrics_dfs)
    statistics_df = statistics_df.reindex(sorted(statistics_df.columns), axis=1)
    statistics_df.to_csv(os.path.join(output_folder, file_name), float_format='%.2f')


if __name__ == "__main__":
    calculate_statistics('metrics.csv', settings['manager'])
    calculate_statistics('overall_metrics.csv', settings['manager'])
