# """
# Metrics for computing evalutation results
# """

# import numpy as np

# class Metric(object):
#     """
#     Compute evaluation result

#     Args:
#         max_label:
#             max label index in the data (0 denoting background)
#         n_runs:
#             number of test runs
#     """
#     def __init__(self, max_label=20, n_runs=None):
#         self.labels = list(range(max_label + 1))  # all class labels
#         self.n_runs = 1 if n_runs is None else n_runs

#         # list of list of array, each array save the TP/FP/FN/TN statistic of a testing sample
#         self.tp_lst = [[] for _ in range(self.n_runs)]
#         self.fp_lst = [[] for _ in range(self.n_runs)]
#         self.fn_lst = [[] for _ in range(self.n_runs)]
#         self.tn_lst = [[] for _ in range(self.n_runs)]

#     def record(self, pred, target, labels=None, n_run=None):
#         """
#         Record the evaluation result for each sample and each class label, including:
#             True Positive, False Positive, False Negative, True Negative

#         Args:
#             pred:
#                 predicted mask array, expected shape is H x W
#             target:
#                 target mask array, expected shape is H x W
#             labels:
#                 only count specific label, used when knowing all possible labels in advance
#         """
#         assert pred.shape == target.shape

#         if self.n_runs == 1:
#             n_run = 0

#         # array to save the TP/FP/FN/TN statistic for each class (plus BG)
#         tp_arr = np.full(len(self.labels), np.nan)
#         fp_arr = np.full(len(self.labels), np.nan)
#         fn_arr = np.full(len(self.labels), np.nan)
#         tn_arr = np.full(len(self.labels), np.nan)

#         if labels is None:
#             labels = self.labels
#         else:
#             labels = [0,] + labels

#         # total number of valid pixels (not 255)
#         total_pixels = np.sum(target != 255)

#         for j, label in enumerate(labels):
#             # Get the location of the pixels that are predicted as class j
#             idx = np.where(np.logical_and(pred == j, target != 255))
#             pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
#             # Get the location of the pixels that are class j in ground truth
#             idx = np.where(target == j)
#             target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

#             if target_idx_j:  # if ground-truth contains this class
#                 tp_arr[label] = len(set.intersection(pred_idx_j, target_idx_j))
#                 fp_arr[label] = len(pred_idx_j - target_idx_j)
#                 fn_arr[label] = len(target_idx_j - pred_idx_j)
#                 # TN = total pixels - TP - FP - FN
#                 tn_arr[label] = total_pixels - tp_arr[label] - fp_arr[label] - fn_arr[label]

#         self.tp_lst[n_run].append(tp_arr)
#         self.fp_lst[n_run].append(fp_arr)
#         self.fn_lst[n_run].append(fn_arr)
#         self.tn_lst[n_run].append(tn_arr)

#     def get_mIoU(self, labels=None, n_run=None):
#         """
#         Compute mean IoU

#         Args:
#             labels:
#                 specify a subset of labels to compute mean IoU, default is using all classes
#         """
#         if labels is None:
#             labels = self.labels
#         # Sum TP, FP, FN statistic of all samples
#         if n_run is None:
#             tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
#                       for run in range(self.n_runs)]
#             fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
#                       for run in range(self.n_runs)]
#             fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
#                       for run in range(self.n_runs)]

#             # Compute mean IoU classwisely
#             # Average across n_runs, then average over classes
#             mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
#                                     for run in range(self.n_runs)])
#             mIoU = mIoU_class.mean(axis=1)

#             return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
#                     mIoU.mean(axis=0), mIoU.std(axis=0))
#         else:
#             tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
#             fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
#             fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)

#             # Compute mean IoU classwisely and average over classes
#             mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
#             mIoU = mIoU_class.mean()

#             return mIoU_class, mIoU

#     def get_mIoU_binary(self, n_run=None):
#         """
#         Compute mean IoU for binary scenario
#         (sum all foreground classes as one class)
#         """
#         # Sum TP, FP, FN statistic of all samples
#         if n_run is None:
#             tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0)
#                       for run in range(self.n_runs)]
#             fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0)
#                       for run in range(self.n_runs)]
#             fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0)
#                       for run in range(self.n_runs)]

#             # Sum over all foreground classes
#             tp_sum = [np.c_[tp_sum[run][0], np.nansum(tp_sum[run][1:])]
#                       for run in range(self.n_runs)]
#             fp_sum = [np.c_[fp_sum[run][0], np.nansum(fp_sum[run][1:])]
#                       for run in range(self.n_runs)]
#             fn_sum = [np.c_[fn_sum[run][0], np.nansum(fn_sum[run][1:])]
#                       for run in range(self.n_runs)]

#             # Compute mean IoU classwisely and average across classes
#             mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
#                                     for run in range(self.n_runs)])
#             mIoU = mIoU_class.mean(axis=1)

#             return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
#                     mIoU.mean(axis=0), mIoU.std(axis=0))
#         else:
#             tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0)
#             fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0)
#             fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0)

#             # Sum over all foreground classes
#             tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
#             fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
#             fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]

#             mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
#             mIoU = mIoU_class.mean()

#             return mIoU_class, mIoU

#     def get_confusion_matrix_components(self, labels=None, n_run=None):
#         """
#         Get TP, TN, FP, FN for each class

#         Args:
#             labels: list of labels to compute metrics for
#             n_run: specific run to get metrics for (None for all runs)

#         Returns:
#             Dictionary with TP, TN, FP, FN arrays for each class
#         """
#         if labels is None:
#             labels = self.labels[1:]  # exclude background (0)

#         if n_run is None:
#             # Aggregate across all runs
#             tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
#                       for run in range(self.n_runs)]
#             fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
#                       for run in range(self.n_runs)]
#             fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
#                       for run in range(self.n_runs)]
#             tn_sum = [np.nansum(np.vstack(self.tn_lst[run]), axis=0).take(labels)
#                       for run in range(self.n_runs)]

#             # Average across runs
#             tp_avg = np.mean(np.array([arr[:len(labels)] for arr in tp_sum]), axis=0)
#             fp_avg = np.mean(np.array([arr[:len(labels)] for arr in fp_sum]), axis=0)
#             fn_avg = np.mean(np.array([arr[:len(labels)] for arr in fn_sum]), axis=0)
#             tn_avg = np.mean(np.array([arr[:len(labels)] for arr in tn_sum]), axis=0)

#             # Calculate std
#             tp_std = np.std(np.array([arr[:len(labels)] for arr in tp_sum]), axis=0)
#             fp_std = np.std(np.array([arr[:len(labels)] for arr in fp_sum]), axis=0)
#             fn_std = np.std(np.array([arr[:len(labels)] for arr in fn_sum]), axis=0)
#             tn_std = np.std(np.array([arr[:len(labels)] for arr in tn_sum]), axis=0)

#             return {
#                 'tp': tp_avg, 'tp_std': tp_std,
#                 'fp': fp_avg, 'fp_std': fp_std,
#                 'fn': fn_avg, 'fn_std': fn_std,
#                 'tn': tn_avg, 'tn_std': tn_std,
#                 'labels': labels
#             }
#         else:
#             tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
#             fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
#             fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)
#             tn_sum = np.nansum(np.vstack(self.tn_lst[n_run]), axis=0).take(labels)

#             return {
#                 'tp': tp_sum, 'fp': fp_sum, 'fn': fn_sum, 'tn': tn_sum,
#                 'labels': labels
#             }

#     def save_per_sample_results(self, save_dir, labels=None, class_names=None):
#         """
#         Save TP, TN, FP, FN for each sample, separated by class (one file per class)

#         Args:
#             save_dir: directory to save Excel files
#             labels: list of labels to save
#             class_names: list of class names corresponding to labels (e.g., ['股骨', '大收肌', ...])

#         Returns:
#             List of saved file paths
#         """
#         import os
#         try:
#             import pandas as pd
#         except ImportError:
#             print("Please install pandas: pip install pandas")
#             return None

#         try:
#             from openpyxl import Workbook
#             from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
#         except ImportError:
#             print("Please install openpyxl: pip install openpyxl")
#             return None

#         os.makedirs(save_dir, exist_ok=True)

#         if labels is None:
#             labels = list(range(1, len(self.labels)))

#         if class_names is None:
#             class_names = [f'Class_{l}' for l in labels]

#         saved_files = []

#         for class_idx, (label, class_name) in enumerate(zip(labels, class_names)):
#             wb = Workbook()
#             ws = wb.active
#             ws.title = class_name

#             # Style definitions
#             header_font = Font(bold=True, color="FFFFFF")
#             header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
#             thin_border = Border(
#                 left=Side(style='thin'),
#                 right=Side(style='thin'),
#                 top=Side(style='thin'),
#                 bottom=Side(style='thin')
#             )

#             # Write headers
#             headers = ['Sample Index', 'Run', 'TP', 'TN', 'FP', 'FN', 'IoU', 'Dice']
#             for col_idx, header in enumerate(headers, 1):
#                 cell = ws.cell(row=1, column=col_idx, value=header)
#                 cell.font = header_font
#                 cell.fill = header_fill
#                 cell.alignment = Alignment(horizontal='center')
#                 cell.border = thin_border

#             # Write data for each sample
#             row_num = 2
#             for n_run in range(self.n_runs):
#                 run_tp = np.vstack(self.tp_lst[n_run])
#                 run_fp = np.vstack(self.fp_lst[n_run])
#                 run_fn = np.vstack(self.fn_lst[n_run])
#                 run_tn = np.vstack(self.tn_lst[n_run])

#                 for sample_idx in range(len(run_tp)):
#                     tp_val = run_tp[sample_idx, label] if not np.isnan(run_tp[sample_idx, label]) else 0
#                     fp_val = run_fp[sample_idx, label] if not np.isnan(run_fp[sample_idx, label]) else 0
#                     fn_val = run_fn[sample_idx, label] if not np.isnan(run_fn[sample_idx, label]) else 0
#                     tn_val = run_tn[sample_idx, label] if not np.isnan(run_tn[sample_idx, label]) else 0

#                     # Calculate IoU and Dice
#                     iou = tp_val / (tp_val + fp_val + fn_val) if (tp_val + fp_val + fn_val) > 0 else 0
#                     dice = 2 * tp_val / (2 * tp_val + fp_val + fn_val) if (2 * tp_val + fp_val + fn_val) > 0 else 0

#                     data = [sample_idx, n_run, int(tp_val), int(tn_val), int(fp_val), int(fn_val),
#                             round(iou, 4), round(dice, 4)]

#                     for col_idx, val in enumerate(data, 1):
#                         cell = ws.cell(row=row_num, column=col_idx, value=val)
#                         cell.border = thin_border
#                         if col_idx > 2:  # Numeric columns
#                             cell.alignment = Alignment(horizontal='center')

#                     row_num += 1

#             # Adjust column widths
#             ws.column_dimensions['A'].width = 12
#             ws.column_dimensions['B'].width = 8
#             ws.column_dimensions['C'].width = 10
#             ws.column_dimensions['D'].width = 10
#             ws.column_dimensions['E'].width = 10
#             ws.column_dimensions['F'].width = 10
#             ws.column_dimensions['G'].width = 10
#             ws.column_dimensions['H'].width = 10

#             # Save file
#             filename = f'{class_name}_per_sample.xlsx'
#             filepath = os.path.join(save_dir, filename)
#             wb.save(filepath)
#             saved_files.append(filepath)
#             print(f"Saved: {filepath}")

#         return saved_files



"""
Metrics for computing evalutation results
"""

import numpy as np

class Metric(object):
    """
    Compute evaluation result

    Args:
        max_label:
            max label index in the data (0 denoting background)
        n_runs:
            number of test runs
    """
    def __init__(self, max_label=20, n_runs=None):
        self.labels = list(range(max_label + 1))  # all class labels
        self.n_runs = 1 if n_runs is None else n_runs

        # list of list of array, each array save the TP/FP/FN/TN statistic of a testing sample
        self.tp_lst = [[] for _ in range(self.n_runs)]
        self.fp_lst = [[] for _ in range(self.n_runs)]
        self.fn_lst = [[] for _ in range(self.n_runs)]
        self.tn_lst = [[] for _ in range(self.n_runs)]
        self.time_lst = [[] for _ in range(self.n_runs)]  # smh0521

    def record(self, pred, target, labels=None, n_run=None, inference_time=None):
        """
        Record the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative, True Negative

        Args:
            pred:
                predicted mask array, expected shape is H x W
            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
            inference_time:  # smh0521
                inference time in seconds for this sample
        """
        assert pred.shape == target.shape

        if self.n_runs == 1:
            n_run = 0

        # array to save the TP/FP/FN/TN statistic for each class (plus BG)
        tp_arr = np.full(len(self.labels), np.nan)
        fp_arr = np.full(len(self.labels), np.nan)
        fn_arr = np.full(len(self.labels), np.nan)
        tn_arr = np.full(len(self.labels), np.nan)

        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels

        # total number of valid pixels (not 255)
        total_pixels = np.sum(target != 255)

        for j, label in enumerate(labels):
            # Get the location of the pixels that are predicted as class j
            idx = np.where(np.logical_and(pred == j, target != 255))
            pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # Get the location of the pixels that are class j in ground truth
            idx = np.where(target == j)
            target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))

            if target_idx_j:  # if ground-truth contains this class
                tp_arr[label] = len(set.intersection(pred_idx_j, target_idx_j))
                fp_arr[label] = len(pred_idx_j - target_idx_j)
                fn_arr[label] = len(target_idx_j - pred_idx_j)
                # TN = total pixels - TP - FP - FN
                tn_arr[label] = total_pixels - tp_arr[label] - fp_arr[label] - fn_arr[label]

        self.tp_lst[n_run].append(tp_arr)
        self.fp_lst[n_run].append(fp_arr)
        self.fn_lst[n_run].append(fn_arr)
        self.tn_lst[n_run].append(tn_arr)
        if inference_time is not None:
            self.time_lst[n_run].append(inference_time)

    def get_mIoU(self, labels=None, n_run=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely
            # Average across n_runs, then average over classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

    def get_mIoU_binary(self, n_run=None):
        """
        Compute mean IoU for binary scenario
        (sum all foreground classes as one class)
        """
        # Sum TP, FP, FN statistic of all samples
        if n_run is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0)
                      for run in range(self.n_runs)]

            # Sum over all foreground classes
            tp_sum = [np.c_[tp_sum[run][0], np.nansum(tp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fp_sum = [np.c_[fp_sum[run][0], np.nansum(fp_sum[run][1:])]
                      for run in range(self.n_runs)]
            fn_sum = [np.c_[fn_sum[run][0], np.nansum(fn_sum[run][1:])]
                      for run in range(self.n_runs)]

            # Compute mean IoU classwisely and average across classes
            mIoU_class = np.vstack([tp_sum[run] / (tp_sum[run] + fp_sum[run] + fn_sum[run])
                                    for run in range(self.n_runs)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0)

            # Sum over all foreground classes
            tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
            fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
            fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]

            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

    def get_confusion_matrix_components(self, labels=None, n_run=None):
        """
        Get TP, TN, FP, FN for each class

        Args:
            labels: list of labels to compute metrics for
            n_run: specific run to get metrics for (None for all runs)

        Returns:
            Dictionary with TP, TN, FP, FN arrays for each class
        """
        if labels is None:
            labels = self.labels[1:]  # exclude background (0)

        if n_run is None:
            # Aggregate across all runs
            tp_sum = [np.nansum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]
            tn_sum = [np.nansum(np.vstack(self.tn_lst[run]), axis=0).take(labels)
                      for run in range(self.n_runs)]

            # Average across runs
            tp_avg = np.mean(np.array([arr[:len(labels)] for arr in tp_sum]), axis=0)
            fp_avg = np.mean(np.array([arr[:len(labels)] for arr in fp_sum]), axis=0)
            fn_avg = np.mean(np.array([arr[:len(labels)] for arr in fn_sum]), axis=0)
            tn_avg = np.mean(np.array([arr[:len(labels)] for arr in tn_sum]), axis=0)

            # Calculate std
            tp_std = np.std(np.array([arr[:len(labels)] for arr in tp_sum]), axis=0)
            fp_std = np.std(np.array([arr[:len(labels)] for arr in fp_sum]), axis=0)
            fn_std = np.std(np.array([arr[:len(labels)] for arr in fn_sum]), axis=0)
            tn_std = np.std(np.array([arr[:len(labels)] for arr in tn_sum]), axis=0)

            return {
                'tp': tp_avg, 'tp_std': tp_std,
                'fp': fp_avg, 'fp_std': fp_std,
                'fn': fn_avg, 'fn_std': fn_std,
                'tn': tn_avg, 'tn_std': tn_std,
                'labels': labels
            }
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)
            tn_sum = np.nansum(np.vstack(self.tn_lst[n_run]), axis=0).take(labels)

            return {
                'tp': tp_sum, 'fp': fp_sum, 'fn': fn_sum, 'tn': tn_sum,
                'labels': labels
            }

    def save_per_sample_results(self, save_dir, labels=None, class_names=None):
        """
        Save TP, TN, FP, FN for each sample, separated by class (one file per class)

        Args:
            save_dir: directory to save Excel files
            labels: list of labels to save
            class_names: list of class names corresponding to labels (e.g., ['股骨', '大收肌', ...])

        Returns:
            List of saved file paths
        """
        import os
        try:
            import pandas as pd
        except ImportError:
            print("Please install pandas: pip install pandas")
            return None

        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        except ImportError:
            print("Please install openpyxl: pip install openpyxl")
            return None

        os.makedirs(save_dir, exist_ok=True)

        if labels is None:
            labels = list(range(1, len(self.labels)))

        if class_names is None:
            class_names = [f'Class_{l}' for l in labels]

        saved_files = []

        for class_idx, (label, class_name) in enumerate(zip(labels, class_names)):
            wb = Workbook()
            ws = wb.active
            ws.title = class_name

            # Style definitions
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

            # Write headers
            headers = ['Sample Index', 'Run', 'TP', 'TN', 'FP', 'FN', 'IoU', 'Dice', 'Inference Time (s)']
            for col_idx, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col_idx, value=header)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')
                cell.border = thin_border

            # Write data for each sample
            row_num = 2
            for n_run in range(self.n_runs):
                run_tp = np.vstack(self.tp_lst[n_run])
                run_fp = np.vstack(self.fp_lst[n_run])
                run_fn = np.vstack(self.fn_lst[n_run])
                run_tn = np.vstack(self.tn_lst[n_run])
                run_times = self.time_lst[n_run] if self.time_lst[n_run] else [None] * len(run_tp)

                for sample_idx in range(len(run_tp)):
                    tp_val = run_tp[sample_idx, label] if not np.isnan(run_tp[sample_idx, label]) else 0
                    fp_val = run_fp[sample_idx, label] if not np.isnan(run_fp[sample_idx, label]) else 0
                    fn_val = run_fn[sample_idx, label] if not np.isnan(run_fn[sample_idx, label]) else 0
                    tn_val = run_tn[sample_idx, label] if not np.isnan(run_tn[sample_idx, label]) else 0

                    # Calculate IoU and Dice
                    iou = tp_val / (tp_val + fp_val + fn_val) if (tp_val + fp_val + fn_val) > 0 else 0
                    dice = 2 * tp_val / (2 * tp_val + fp_val + fn_val) if (2 * tp_val + fp_val + fn_val) > 0 else 0

                    # smh0521: inference time for this sample (same for all classes)
                    inf_time = round(run_times[sample_idx], 4) if sample_idx < len(run_times) and run_times[sample_idx] is not None else None

                    data = [sample_idx, n_run, int(tp_val), int(tn_val), int(fp_val), int(fn_val),
                            round(iou, 4), round(dice, 4), inf_time]

                    for col_idx, val in enumerate(data, 1):
                        cell = ws.cell(row=row_num, column=col_idx, value=val)
                        cell.border = thin_border
                        if col_idx > 2:  # Numeric columns
                            cell.alignment = Alignment(horizontal='center')

                    row_num += 1

            # Adjust column widths
            ws.column_dimensions['A'].width = 12
            ws.column_dimensions['B'].width = 8
            ws.column_dimensions['C'].width = 10
            ws.column_dimensions['D'].width = 10
            ws.column_dimensions['E'].width = 10
            ws.column_dimensions['F'].width = 10
            ws.column_dimensions['G'].width = 10
            ws.column_dimensions['H'].width = 10
            ws.column_dimensions['I'].width = 18  # smh0521

            # Save file
            filename = f'{class_name}_per_sample.xlsx'
            filepath = os.path.join(save_dir, filename)
            wb.save(filepath)
            saved_files.append(filepath)
            print(f"Saved: {filepath}")

        return saved_files
