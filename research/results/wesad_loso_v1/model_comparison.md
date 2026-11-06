# Complete WESAD model comparison

Each row summarizes the same 15 held-out participants with equal participant weights. Macro F1 includes the between-participant sample SD. The binary sensitivity and specificity columns refer to the stress class. Multiclass columns use macro one-vs-rest averaging. Size is the median fold checkpoint size in KiB. Parameter counts apply to linear and neural models.

See the [interpretation](README.md), [all metrics and confidence intervals](all_models.csv) and [per-participant results](per_subject.csv).

## Zero-calibration primary candidates

| Experiment | Wrist inputs | Accuracy | Macro F1 ± SD | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neural/cnn_bilstm_zero` | eda | 0.782 | 0.746 ± 0.199 | 0.777 | 0.741 | 0.812 | 0.872 | 0.848 | 0.177 | 4,594 | 24.5 |
| `neural/cnn_tcn_attention_zero` | eda | 0.828 | 0.806 ± 0.157 | 0.829 | 0.826 | 0.832 | 0.945 | 0.923 | 0.137 | 3,827 | 22.5 |
| `neural/cnn_transformer_zero` | eda | 0.799 | 0.763 ± 0.210 | 0.791 | 0.746 | 0.836 | 0.909 | 0.905 | 0.173 | 5,154 | 39.1 |
| `neural/hybrid_features_zero` | eda | 0.803 | 0.777 ± 0.186 | 0.805 | 0.805 | 0.805 | 0.944 | 0.911 | 0.151 | 5,347 | 30.7 |
| `neural/multitcn_gate_zero` | eda | 0.793 | 0.760 ± 0.214 | 0.797 | 0.804 | 0.790 | 0.933 | 0.887 | 0.170 | 5,187 | 29.6 |
| `neural/multitcn_gru_zero` | eda | 0.782 | 0.749 ± 0.231 | 0.787 | 0.798 | 0.775 | 0.915 | 0.882 | 0.179 | 6,530 | 34.8 |
| `neural/multitcn_transformer_zero` | eda | 0.783 | 0.744 ± 0.242 | 0.785 | 0.780 | 0.790 | 0.938 | 0.907 | 0.162 | 7,122 | 50.4 |
| `neural/pg_eda_zero` | eda | 0.794 | 0.762 ± 0.223 | 0.793 | 0.782 | 0.805 | 0.932 | 0.904 | 0.157 | 8,843 | 49.8 |
| `neural/pg_fullwrist_zero` | eda+acc+temp+bvp | 0.842 | 0.820 ± 0.213 | 0.850 | 0.883 | 0.817 | 0.985 | 0.975 | 0.128 | 9,723 | 54.1 |
| `neural/pg_temp_zero` | eda+acc+temp | 0.856 | 0.836 ± 0.180 | 0.862 | 0.885 | 0.839 | 0.972 | 0.967 | 0.118 | 9,435 | 52.8 |
| `neural/pg_zero` | eda+acc | 0.858 | 0.841 ± 0.123 | 0.850 | 0.806 | 0.893 | 0.968 | 0.957 | 0.109 | 9,323 | 52.1 |
| `neural/physics_zero` | eda+acc | 0.813 | 0.784 ± 0.218 | 0.816 | 0.823 | 0.809 | 0.946 | 0.913 | 0.145 | 9,382 | 53.9 |
| `neural/reliable_zero` | eda+acc | 0.854 | 0.838 ± 0.181 | 0.853 | 0.841 | 0.864 | 0.972 | 0.962 | 0.115 | 8,275 | 46.2 |
| `neural/tcn_attention_zero` | eda | 0.783 | 0.756 ± 0.195 | 0.789 | 0.805 | 0.772 | 0.933 | 0.900 | 0.162 | 4,131 | 24.6 |
| `neural/tonic_attention_zero` | eda | 0.779 | 0.747 ± 0.206 | 0.778 | 0.763 | 0.793 | 0.931 | 0.904 | 0.173 | 9,619 | 52.3 |
| `neural/tonic_phasic_zero` | eda | 0.805 | 0.774 ± 0.214 | 0.805 | 0.794 | 0.815 | 0.940 | 0.914 | 0.155 | 8,531 | 47.2 |
| `neural/contrast_tonic_zero` | eda | 0.789 | 0.756 ± 0.224 | 0.789 | 0.782 | 0.796 | 0.927 | 0.910 | 0.160 | 9,619 | 52.3 |
| `neural/masked_multitcn_zero` | eda | 0.770 | 0.725 ± 0.272 | 0.775 | 0.786 | 0.764 | 0.934 | 0.902 | 0.171 | 4,898 | 27.6 |
| `neural/bigru_zero` | eda | 0.777 | 0.737 ± 0.247 | 0.776 | 0.763 | 0.789 | 0.888 | 0.869 | 0.183 | 2,098 | 14.1 |
| `neural/bilstm_zero` | eda | 0.786 | 0.752 ± 0.205 | 0.782 | 0.752 | 0.811 | 0.939 | 0.909 | 0.155 | 2,514 | 15.6 |
| `neural/cnn_zero` | eda | 0.772 | 0.735 ± 0.221 | 0.769 | 0.748 | 0.791 | 0.932 | 0.904 | 0.171 | 2,930 | 16.4 |
| `neural/gru_zero` | eda | 0.777 | 0.743 ± 0.217 | 0.776 | 0.763 | 0.788 | 0.938 | 0.928 | 0.164 | 2,482 | 14.6 |
| `neural/lstm_zero` | eda | 0.792 | 0.759 ± 0.199 | 0.788 | 0.766 | 0.809 | 0.922 | 0.901 | 0.159 | 3,026 | 16.8 |
| `neural/multitcn_zero` | eda | 0.774 | 0.737 ± 0.238 | 0.780 | 0.795 | 0.764 | 0.939 | 0.910 | 0.168 | 4,898 | 27.6 |
| `neural/tcn_zero` | eda | 0.783 | 0.748 ± 0.226 | 0.785 | 0.782 | 0.787 | 0.945 | 0.920 | 0.163 | 4,114 | 24.0 |
| `neural/transformer_zero` | eda | 0.770 | 0.735 ± 0.200 | 0.768 | 0.749 | 0.787 | 0.854 | 0.814 | 0.184 | 3,074 | 30.0 |
| `classical/lightgbm__eda-acc-temp-bvp__zero` | eda+acc+temp+bvp | 0.891 | 0.874 ± 0.178 | 0.889 | 0.884 | 0.893 | 0.984 | 0.980 | 0.093 | — | 31.6 |
| `classical/lightgbm__eda-acc-temp__zero` | eda+acc+temp | 0.906 | 0.887 ± 0.162 | 0.899 | 0.866 | 0.931 | 0.968 | 0.962 | 0.090 | — | 31.3 |
| `classical/lightgbm__eda-acc__zero` | eda+acc | 0.875 | 0.856 ± 0.166 | 0.870 | 0.840 | 0.899 | 0.970 | 0.959 | 0.100 | — | 60.6 |
| `classical/lightgbm__eda__zero` | eda | 0.842 | 0.813 ± 0.194 | 0.835 | 0.798 | 0.873 | 0.941 | 0.915 | 0.131 | — | 30.9 |
| `classical/logistic__eda-acc-temp-bvp__zero` | eda+acc+temp+bvp | 0.903 | 0.897 ± 0.109 | 0.904 | 0.910 | 0.898 | 0.993 | 0.988 | 0.080 | 40 | 2.1 |
| `classical/logistic__eda-acc-temp__zero` | eda+acc+temp | 0.871 | 0.860 ± 0.124 | 0.869 | 0.853 | 0.885 | 0.970 | 0.964 | 0.115 | 34 | 2.0 |
| `classical/logistic__eda-acc__zero` | eda+acc | 0.886 | 0.873 ± 0.117 | 0.877 | 0.829 | 0.924 | 0.966 | 0.954 | 0.112 | 31 | 1.9 |
| `classical/logistic__eda__zero` | eda | 0.834 | 0.808 ± 0.163 | 0.823 | 0.768 | 0.879 | 0.878 | 0.872 | 0.154 | 21 | 1.6 |
| `classical/random_forest__eda-acc-temp-bvp__zero` | eda+acc+temp+bvp | 0.877 | 0.859 ± 0.201 | 0.876 | 0.869 | 0.882 | 0.966 | 0.964 | 0.105 | — | 181.9 |
| `classical/random_forest__eda-acc-temp__zero` | eda+acc+temp | 0.907 | 0.887 ± 0.173 | 0.898 | 0.858 | 0.939 | 0.963 | 0.964 | 0.090 | — | 209.1 |
| `classical/random_forest__eda-acc__zero` | eda+acc | 0.862 | 0.836 ± 0.188 | 0.853 | 0.809 | 0.897 | 0.966 | 0.949 | 0.115 | — | 253.1 |
| `classical/random_forest__eda__zero` | eda | 0.851 | 0.825 ± 0.170 | 0.842 | 0.798 | 0.887 | 0.934 | 0.910 | 0.125 | — | 281.6 |
| `classical/rbf_svm__eda-acc-temp-bvp__zero` | eda+acc+temp+bvp | 0.913 | 0.908 ± 0.078 | 0.912 | 0.908 | 0.916 | 0.992 | 0.982 | 0.067 | — | 53.8 |
| `classical/rbf_svm__eda-acc-temp__zero` | eda+acc+temp | 0.906 | 0.899 ± 0.095 | 0.902 | 0.880 | 0.924 | 0.968 | 0.961 | 0.082 | — | 57.0 |
| `classical/rbf_svm__eda-acc__zero` | eda+acc | 0.885 | 0.875 ± 0.103 | 0.880 | 0.853 | 0.906 | 0.968 | 0.950 | 0.094 | — | 57.3 |
| `classical/rbf_svm__eda__zero` | eda | 0.810 | 0.788 ± 0.168 | 0.806 | 0.782 | 0.830 | 0.885 | 0.848 | 0.149 | — | 42.8 |
| `classical/xgboost__eda-acc-temp-bvp__zero` | eda+acc+temp+bvp | 0.888 | 0.868 ± 0.203 | 0.889 | 0.898 | 0.880 | 0.977 | 0.972 | 0.090 | — | 11.7 |
| `classical/xgboost__eda-acc-temp__zero` | eda+acc+temp | 0.894 | 0.875 ± 0.167 | 0.889 | 0.865 | 0.913 | 0.954 | 0.951 | 0.094 | — | 11.7 |
| `classical/xgboost__eda-acc__zero` | eda+acc | 0.873 | 0.855 ± 0.171 | 0.871 | 0.854 | 0.887 | 0.960 | 0.950 | 0.108 | — | 28.5 |
| `classical/xgboost__eda__zero` | eda | 0.832 | 0.810 ± 0.179 | 0.831 | 0.820 | 0.842 | 0.930 | 0.906 | 0.126 | — | 29.3 |

## Resting-calibrated primary candidates

| Experiment | Wrist inputs | Accuracy | Macro F1 ± SD | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neural/cnn_bilstm_rest` | eda | 0.856 | 0.821 ± 0.214 | 0.846 | 0.791 | 0.900 | 0.952 | 0.943 | 0.136 | 4,594 | 24.5 |
| `neural/cnn_tcn_attention_rest` | eda | 0.880 | 0.848 ± 0.197 | 0.865 | 0.792 | 0.938 | 0.965 | 0.957 | 0.117 | 3,827 | 22.5 |
| `neural/cnn_transformer_rest` | eda | 0.848 | 0.817 ± 0.208 | 0.845 | 0.822 | 0.868 | 0.955 | 0.937 | 0.143 | 5,154 | 39.1 |
| `neural/hybrid_features_rest` | eda | 0.866 | 0.832 ± 0.222 | 0.861 | 0.830 | 0.891 | 0.954 | 0.945 | 0.133 | 5,347 | 30.7 |
| `neural/multitcn_gate_rest` | eda | 0.839 | 0.803 ± 0.220 | 0.833 | 0.804 | 0.863 | 0.944 | 0.937 | 0.150 | 5,187 | 29.6 |
| `neural/multitcn_gru_rest` | eda | 0.855 | 0.821 ± 0.216 | 0.849 | 0.820 | 0.878 | 0.939 | 0.924 | 0.140 | 6,530 | 34.8 |
| `neural/multitcn_transformer_rest` | eda | 0.850 | 0.816 ± 0.206 | 0.843 | 0.809 | 0.877 | 0.961 | 0.947 | 0.140 | 7,122 | 50.4 |
| `neural/pg_eda_rest` | eda | 0.838 | 0.807 ± 0.193 | 0.832 | 0.804 | 0.861 | 0.960 | 0.946 | 0.143 | 8,843 | 49.8 |
| `neural/pg_fullwrist_rest` | eda+acc+temp+bvp | 0.876 | 0.847 ± 0.188 | 0.867 | 0.827 | 0.907 | 0.982 | 0.980 | 0.101 | 9,723 | 54.1 |
| `neural/pg_rest` | eda+acc | 0.908 | 0.887 ± 0.159 | 0.897 | 0.842 | 0.951 | 0.963 | 0.956 | 0.090 | 9,323 | 52.1 |
| `neural/pg_temp_rest` | eda+acc+temp | 0.850 | 0.817 ± 0.191 | 0.840 | 0.793 | 0.886 | 0.949 | 0.929 | 0.125 | 9,435 | 52.8 |
| `neural/physics_rest` | eda+acc | 0.892 | 0.864 ± 0.193 | 0.883 | 0.838 | 0.928 | 0.955 | 0.933 | 0.086 | 9,382 | 53.9 |
| `neural/reliable_rest` | eda+acc | 0.909 | 0.899 ± 0.130 | 0.905 | 0.879 | 0.930 | 0.986 | 0.979 | 0.076 | 8,275 | 46.2 |
| `neural/tcn_attention_rest` | eda | 0.858 | 0.832 ± 0.184 | 0.853 | 0.827 | 0.880 | 0.966 | 0.955 | 0.121 | 4,131 | 24.6 |
| `neural/tonic_attention_rest` | eda | 0.826 | 0.778 ± 0.276 | 0.821 | 0.793 | 0.848 | 0.955 | 0.941 | 0.173 | 9,619 | 52.3 |
| `neural/tonic_phasic_rest` | eda | 0.843 | 0.803 ± 0.240 | 0.838 | 0.810 | 0.866 | 0.940 | 0.935 | 0.151 | 8,531 | 47.2 |
| `neural/contrast_tonic_rest` | eda | 0.806 | 0.759 ± 0.233 | 0.796 | 0.750 | 0.843 | 0.935 | 0.924 | 0.171 | 9,619 | 52.3 |
| `neural/masked_multitcn_rest` | eda | 0.831 | 0.800 ± 0.191 | 0.827 | 0.808 | 0.846 | 0.940 | 0.925 | 0.145 | 4,898 | 27.6 |
| `neural/bigru_rest` | eda | 0.832 | 0.792 ± 0.238 | 0.829 | 0.808 | 0.850 | 0.950 | 0.934 | 0.155 | 2,098 | 14.1 |
| `neural/bilstm_rest` | eda | 0.881 | 0.856 ± 0.175 | 0.872 | 0.832 | 0.912 | 0.974 | 0.960 | 0.114 | 2,514 | 15.6 |
| `neural/cnn_rest` | eda | 0.859 | 0.829 ± 0.194 | 0.853 | 0.823 | 0.882 | 0.962 | 0.948 | 0.133 | 2,930 | 16.4 |
| `neural/gru_rest` | eda | 0.840 | 0.807 ± 0.209 | 0.836 | 0.813 | 0.859 | 0.947 | 0.930 | 0.146 | 2,482 | 14.6 |
| `neural/lstm_rest` | eda | 0.838 | 0.806 ± 0.209 | 0.835 | 0.818 | 0.852 | 0.955 | 0.946 | 0.135 | 3,026 | 16.8 |
| `neural/multitcn_rest` | eda | 0.849 | 0.814 ± 0.205 | 0.841 | 0.799 | 0.882 | 0.958 | 0.949 | 0.139 | 4,898 | 27.6 |
| `neural/tcn_rest` | eda | 0.843 | 0.814 ± 0.202 | 0.842 | 0.830 | 0.853 | 0.965 | 0.954 | 0.138 | 4,114 | 24.0 |
| `neural/transformer_rest` | eda | 0.858 | 0.824 ± 0.215 | 0.849 | 0.805 | 0.894 | 0.926 | 0.900 | 0.136 | 3,074 | 30.0 |
| `classical/lightgbm__eda-acc-temp-bvp__rest` | eda+acc+temp+bvp | 0.855 | 0.819 ± 0.232 | 0.842 | 0.792 | 0.893 | 0.953 | 0.953 | 0.117 | — | 30.9 |
| `classical/lightgbm__eda-acc-temp__rest` | eda+acc+temp | 0.880 | 0.849 ± 0.201 | 0.867 | 0.815 | 0.919 | 0.947 | 0.948 | 0.100 | — | 30.8 |
| `classical/lightgbm__eda-acc__rest` | eda+acc | 0.888 | 0.856 ± 0.197 | 0.872 | 0.804 | 0.940 | 0.967 | 0.954 | 0.105 | — | 30.9 |
| `classical/lightgbm__eda__rest` | eda | 0.805 | 0.767 ± 0.221 | 0.803 | 0.792 | 0.815 | 0.921 | 0.900 | 0.152 | — | 30.4 |
| `classical/logistic__eda-acc-temp-bvp__rest` | eda+acc+temp+bvp | 0.908 | 0.900 ± 0.101 | 0.904 | 0.886 | 0.922 | 0.987 | 0.984 | 0.075 | 40 | 2.2 |
| `classical/logistic__eda-acc-temp__rest` | eda+acc+temp | 0.903 | 0.890 ± 0.148 | 0.899 | 0.872 | 0.925 | 0.956 | 0.952 | 0.103 | 34 | 2.0 |
| `classical/logistic__eda-acc__rest` | eda+acc | 0.904 | 0.889 ± 0.144 | 0.896 | 0.857 | 0.936 | 0.955 | 0.947 | 0.092 | 31 | 1.9 |
| `classical/logistic__eda__rest` | eda | 0.885 | 0.856 ± 0.188 | 0.870 | 0.801 | 0.940 | 0.941 | 0.932 | 0.131 | 21 | 1.7 |
| `classical/random_forest__eda-acc-temp-bvp__rest` | eda+acc+temp+bvp | 0.888 | 0.863 ± 0.208 | 0.881 | 0.853 | 0.910 | 0.981 | 0.978 | 0.094 | — | 151.4 |
| `classical/random_forest__eda-acc-temp__rest` | eda+acc+temp | 0.905 | 0.879 ± 0.187 | 0.893 | 0.837 | 0.949 | 0.966 | 0.962 | 0.095 | — | 176.3 |
| `classical/random_forest__eda-acc__rest` | eda+acc | 0.887 | 0.857 ± 0.196 | 0.876 | 0.826 | 0.927 | 0.970 | 0.963 | 0.106 | — | 205.8 |
| `classical/random_forest__eda__rest` | eda | 0.858 | 0.823 ± 0.207 | 0.851 | 0.814 | 0.887 | 0.937 | 0.922 | 0.129 | — | 229.6 |
| `classical/rbf_svm__eda-acc-temp-bvp__rest` | eda+acc+temp+bvp | 0.917 | 0.909 ± 0.107 | 0.913 | 0.890 | 0.935 | 0.991 | 0.984 | 0.065 | — | 47.4 |
| `classical/rbf_svm__eda-acc-temp__rest` | eda+acc+temp | 0.917 | 0.903 ± 0.143 | 0.907 | 0.861 | 0.954 | 0.965 | 0.958 | 0.077 | — | 49.6 |
| `classical/rbf_svm__eda-acc__rest` | eda+acc | 0.914 | 0.900 ± 0.139 | 0.904 | 0.856 | 0.952 | 0.977 | 0.964 | 0.079 | — | 46.0 |
| `classical/rbf_svm__eda__rest` | eda | 0.898 | 0.868 ± 0.200 | 0.883 | 0.813 | 0.954 | 0.963 | 0.948 | 0.102 | — | 36.2 |
| `classical/xgboost__eda-acc-temp-bvp__rest` | eda+acc+temp+bvp | 0.863 | 0.828 ± 0.233 | 0.854 | 0.820 | 0.888 | 0.967 | 0.967 | 0.111 | — | 11.7 |
| `classical/xgboost__eda-acc-temp__rest` | eda+acc+temp | 0.892 | 0.862 ± 0.196 | 0.877 | 0.815 | 0.939 | 0.952 | 0.951 | 0.095 | — | 11.5 |
| `classical/xgboost__eda-acc__rest` | eda+acc | 0.866 | 0.827 ± 0.217 | 0.851 | 0.786 | 0.917 | 0.970 | 0.956 | 0.113 | — | 11.4 |
| `classical/xgboost__eda__rest` | eda | 0.801 | 0.760 ± 0.226 | 0.798 | 0.784 | 0.813 | 0.952 | 0.944 | 0.164 | — | 11.4 |

## Ablations and budget sensitivities

| Experiment | Wrist inputs | Accuracy | Macro F1 ± SD | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neural/pg_bigru_zero` | eda+acc | 0.813 | 0.781 ± 0.206 | 0.807 | 0.770 | 0.844 | 0.931 | 0.913 | 0.151 | 10,571 | 58.9 |
| `neural/pg_concat_zero` | eda+acc | 0.822 | 0.792 ± 0.221 | 0.819 | 0.796 | 0.841 | 0.956 | 0.937 | 0.141 | 10,099 | 55.1 |
| `neural/pg_cosine_zero` | eda+acc | 0.824 | 0.802 ± 0.158 | 0.817 | 0.783 | 0.852 | 0.963 | 0.950 | 0.136 | 9,323 | 52.2 |
| `neural/pg_cross_zero` | eda+acc | 0.856 | 0.835 ± 0.176 | 0.853 | 0.833 | 0.873 | 0.965 | 0.957 | 0.114 | 11,187 | 60.2 |
| `neural/pg_cv_physics_zero` | eda+acc | 0.828 | 0.807 ± 0.190 | 0.833 | 0.849 | 0.816 | 0.948 | 0.924 | 0.130 | 9,382 | 53.9 |
| `neural/pg_cvxeda_zero` | eda+acc | 0.866 | 0.852 ± 0.138 | 0.863 | 0.845 | 0.882 | 0.968 | 0.952 | 0.108 | 9,323 | 52.1 |
| `neural/pg_focal_zero` | eda+acc | 0.842 | 0.816 ± 0.179 | 0.837 | 0.809 | 0.865 | 0.964 | 0.942 | 0.121 | 9,323 | 52.2 |
| `neural/pg_fullwrist_long_zero` | eda+acc+temp+bvp | 0.858 | 0.837 ± 0.201 | 0.864 | 0.893 | 0.836 | 0.984 | 0.974 | 0.121 | 9,723 | 54.2 |
| `neural/pg_gated_zero` | eda+acc | 0.813 | 0.780 ± 0.206 | 0.809 | 0.780 | 0.837 | 0.953 | 0.933 | 0.151 | 9,319 | 52.1 |
| `neural/pg_gru_cross_zero` | eda+acc | 0.896 | 0.886 ± 0.105 | 0.889 | 0.857 | 0.922 | 0.980 | 0.973 | 0.083 | 12,819 | 67.5 |
| `neural/pg_gru_zero` | eda+acc | 0.859 | 0.835 ± 0.164 | 0.852 | 0.810 | 0.893 | 0.935 | 0.926 | 0.122 | 10,955 | 59.4 |
| `neural/pg_intrinsic_quality_zero` | eda+acc | 0.860 | 0.843 ± 0.121 | 0.851 | 0.807 | 0.895 | 0.968 | 0.957 | 0.111 | 9,323 | 52.2 |
| `neural/pg_long_zero` | eda+acc | 0.881 | 0.863 ± 0.119 | 0.867 | 0.800 | 0.934 | 0.971 | 0.960 | 0.093 | 9,323 | 52.2 |
| `neural/pg_no_components_no_hand_zero` | eda+acc | 0.725 | 0.659 ± 0.302 | 0.729 | 0.750 | 0.708 | 0.904 | 0.885 | 0.218 | 8,361 | 47.3 |
| `neural/pg_no_hand_rest` | eda+acc | 0.856 | 0.821 ± 0.200 | 0.846 | 0.795 | 0.897 | 0.958 | 0.954 | 0.123 | 8,681 | 48.7 |
| `neural/pg_no_hand_zero` | eda+acc | 0.810 | 0.782 ± 0.181 | 0.802 | 0.761 | 0.843 | 0.957 | 0.947 | 0.159 | 8,681 | 48.7 |
| `neural/pg_no_relative_no_hand_rest` | eda+acc | 0.812 | 0.780 ± 0.216 | 0.809 | 0.787 | 0.830 | 0.913 | 0.904 | 0.162 | 8,553 | 48.1 |
| `neural/pg_single_scale_zero` | eda+acc | 0.812 | 0.790 ± 0.177 | 0.814 | 0.814 | 0.815 | 0.944 | 0.935 | 0.144 | 8,539 | 48.6 |
| `neural/pg_transformer_zero` | eda+acc | 0.833 | 0.808 ± 0.181 | 0.829 | 0.804 | 0.855 | 0.956 | 0.940 | 0.122 | 11,547 | 75.0 |
| `neural/pg_weighted_zero` | eda+acc | 0.807 | 0.780 ± 0.197 | 0.806 | 0.790 | 0.821 | 0.955 | 0.928 | 0.149 | 9,063 | 50.9 |
| `neural/physics_no_aux_zero` | eda+acc | 0.814 | 0.784 ± 0.219 | 0.817 | 0.825 | 0.809 | 0.946 | 0.913 | 0.145 | 9,382 | 54.0 |
| `neural/multitcn_long_zero` | eda | 0.762 | 0.721 ± 0.247 | 0.768 | 0.781 | 0.755 | 0.930 | 0.904 | 0.177 | 4,898 | 27.6 |
| `classical/ablation__lightgbm__cvxeda` | eda | 0.842 | 0.822 ± 0.177 | 0.840 | 0.831 | 0.850 | 0.915 | 0.875 | 0.137 | — | 30.7 |
| `classical/ablation__logistic__cvxeda` | eda | 0.823 | 0.801 ± 0.141 | 0.813 | 0.767 | 0.860 | 0.906 | 0.906 | 0.152 | 21 | 1.6 |
| `classical/ablation__random_forest__cvxeda` | eda | 0.840 | 0.820 ± 0.165 | 0.838 | 0.827 | 0.850 | 0.923 | 0.895 | 0.126 | — | 296.2 |
| `classical/ablation__rbf_svm__cvxeda` | eda | 0.752 | 0.693 ± 0.173 | 0.719 | 0.563 | 0.874 | 0.892 | 0.862 | 0.188 | — | 62.8 |
| `classical/ablation__rf__calibration-rolling` | eda | 0.849 | 0.823 ± 0.175 | 0.843 | 0.810 | 0.876 | 0.931 | 0.911 | 0.124 | — | 262.2 |
| `classical/ablation__rf__decomposition-choi` | eda | 0.835 | 0.810 ± 0.181 | 0.830 | 0.803 | 0.857 | 0.926 | 0.905 | 0.132 | — | 286.4 |
| `classical/ablation__rf__decomposition-none` | eda | 0.821 | 0.801 ± 0.158 | 0.814 | 0.782 | 0.846 | 0.912 | 0.883 | 0.132 | — | 218.0 |
| `classical/ablation__rf__decomposition-sparse` | eda | 0.834 | 0.818 ± 0.144 | 0.835 | 0.838 | 0.832 | 0.927 | 0.904 | 0.123 | — | 303.4 |
| `classical/ablation__rf__eda-acc__quality-eda` | eda+acc | 0.863 | 0.838 ± 0.184 | 0.855 | 0.815 | 0.895 | 0.964 | 0.946 | 0.113 | — | 250.8 |
| `classical/ablation__rf__eda-acc__quality-off` | eda+acc | 0.869 | 0.845 ± 0.177 | 0.862 | 0.823 | 0.900 | 0.966 | 0.951 | 0.110 | — | 260.3 |
| `classical/ablation__rf__filtering-1hz` | eda | 0.846 | 0.818 ± 0.179 | 0.837 | 0.790 | 0.884 | 0.933 | 0.907 | 0.128 | — | 294.4 |
| `classical/ablation__rf__filtering-raw` | eda | 0.829 | 0.805 ± 0.167 | 0.822 | 0.786 | 0.859 | 0.932 | 0.907 | 0.133 | — | 301.5 |
| `classical/ablation__rf__purity-1.0` | eda | 0.858 | 0.833 ± 0.167 | 0.849 | 0.802 | 0.895 | 0.927 | 0.907 | 0.122 | — | 290.6 |
| `classical/ablation__rf__seconds-30` | eda | 0.832 | 0.808 ± 0.165 | 0.825 | 0.786 | 0.863 | 0.923 | 0.899 | 0.128 | — | 361.1 |
| `classical/ablation__rf__seconds-90` | eda | 0.860 | 0.835 ± 0.185 | 0.854 | 0.815 | 0.892 | 0.945 | 0.924 | 0.122 | — | 187.0 |
| `classical/ablation__rf__stride-30` | eda | 0.840 | 0.815 ± 0.180 | 0.836 | 0.808 | 0.864 | 0.927 | 0.904 | 0.134 | — | 181.6 |
| `classical/ablation__xgboost__cvxeda` | eda | 0.845 | 0.826 ± 0.175 | 0.846 | 0.848 | 0.844 | 0.919 | 0.894 | 0.128 | — | 11.3 |

## Context-only controls

| Experiment | Wrist inputs | Accuracy | Macro F1 ± SD | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `classical/control__rf__acc` | acc | 0.867 | 0.850 ± 0.088 | 0.851 | 0.778 | 0.924 | 0.947 | 0.929 | 0.103 | — | 338.5 |
| `classical/control__rf__temp` | temp | 0.753 | 0.735 ± 0.247 | 0.757 | 0.774 | 0.741 | 0.832 | 0.805 | 0.176 | — | 299.8 |

## Three- and four-condition experiments

| Experiment | Wrist inputs | Accuracy | Macro F1 ± SD | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neural/pg_3class_zero` | eda+acc | 0.644 | 0.612 ± 0.107 | 0.665 | 0.665 | 0.832 | 0.878 | 0.756 | 0.467 | 9,340 | 52.2 |
| `classical/secondary3__lightgbm` | eda | 0.646 | 0.550 ± 0.207 | 0.587 | 0.587 | 0.813 | 0.800 | 0.688 | 0.496 | — | 82.0 |
| `classical/secondary3__logistic` | eda | 0.655 | 0.552 ± 0.153 | 0.579 | 0.579 | 0.814 | 0.807 | 0.664 | 0.553 | 63 | 2.0 |
| `classical/secondary3__random_forest` | eda | 0.636 | 0.534 ± 0.165 | 0.569 | 0.569 | 0.805 | 0.795 | 0.673 | 0.495 | — | 742.2 |
| `classical/secondary3__rbf_svm` | eda | 0.660 | 0.517 ± 0.131 | 0.543 | 0.543 | 0.805 | 0.772 | 0.629 | 0.517 | — | 89.3 |
| `classical/secondary3__xgboost` | eda | 0.613 | 0.524 ± 0.210 | 0.559 | 0.559 | 0.796 | 0.789 | 0.679 | 0.518 | — | 27.2 |
| `neural/pg_4class_zero` | eda+acc | 0.630 | 0.575 ± 0.109 | 0.600 | 0.600 | 0.879 | 0.839 | 0.667 | 0.528 | 9,357 | 52.2 |
| `classical/secondary4__lightgbm` | eda | 0.532 | 0.455 ± 0.174 | 0.500 | 0.500 | 0.839 | 0.773 | 0.586 | 0.609 | — | 109.5 |
| `classical/secondary4__logistic` | eda | 0.581 | 0.511 ± 0.137 | 0.547 | 0.547 | 0.859 | 0.785 | 0.597 | 0.646 | 84 | 2.1 |
| `classical/secondary4__random_forest` | eda | 0.518 | 0.443 ± 0.134 | 0.475 | 0.475 | 0.829 | 0.755 | 0.550 | 0.625 | — | 775.8 |
| `classical/secondary4__rbf_svm` | eda | 0.416 | 0.328 ± 0.101 | 0.421 | 0.421 | 0.804 | 0.714 | 0.514 | 0.688 | — | 126.4 |
| `classical/secondary4__xgboost` | eda | 0.548 | 0.480 ± 0.179 | 0.517 | 0.517 | 0.843 | 0.779 | 0.590 | 0.596 | — | 35.2 |

## Selection repeated within each outer fold

| Experiment | Wrist inputs | Accuracy | Macro F1 ± SD | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | KiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nested/rest_highest_inner` | varies | 0.848 | 0.818 ± 0.204 | 0.838 | 0.800 | 0.877 | 0.972 | 0.970 | 0.129 | — | — |
| `nested/rest_lightweight` | varies | 0.847 | 0.819 ± 0.184 | 0.838 | 0.792 | 0.883 | 0.973 | 0.960 | 0.135 | — | — |
| `nested/zero_highest_inner` | varies | 0.856 | 0.835 ± 0.182 | 0.854 | 0.846 | 0.863 | 0.971 | 0.956 | 0.112 | — | — |
| `nested/zero_lightweight` | varies | 0.876 | 0.855 ± 0.169 | 0.872 | 0.849 | 0.894 | 0.970 | 0.952 | 0.106 | — | — |
