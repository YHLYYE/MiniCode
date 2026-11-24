# MiniCode 真实编码任务评测

- Run ID: `20261124T183547Z`
- Model: `deepseek-chat`
- Started: 2026-11-24T18:35:47.941093+00:00
- Repeats per task: 3
- Attempted: 60
- Passed: 54/60
- False completions: 2/60

| Task | Repeat | Passed | Claimed | False completion | Turns | Tokens | Cost | Seconds | Tools | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clamp_bounds | 1 | yes | yes | no | 6 | 19317 | $0.0055 | 13.4 | 7 | 0 |
| clamp_bounds | 2 | yes | yes | no | 5 | 15591 | $0.0045 | 6.5 | 6 | 0 |
| clamp_bounds | 3 | yes | yes | no | 6 | 18219 | $0.0052 | 8.0 | 5 | 0 |
| paginate_last_item | 1 | yes | yes | no | 5 | 15782 | $0.0045 | 7.7 | 6 | 0 |
| paginate_last_item | 2 | yes | yes | no | 5 | 15642 | $0.0045 | 5.8 | 6 | 0 |
| paginate_last_item | 3 | yes | yes | no | 5 | 15667 | $0.0045 | 6.7 | 6 | 0 |
| parse_compound_duration | 1 | yes | yes | no | 10 | 37661 | $0.0108 | 19.0 | 11 | 0 |
| parse_compound_duration | 2 | yes | no | no | 10 | 38769 | $0.0111 | 16.8 | 11 | 1 |
| parse_compound_duration | 3 | yes | no | no | 10 | 41716 | $0.0120 | 20.0 | 12 | 0 |
| merge_intervals | 1 | no | yes | yes | 7 | 29114 | $0.0083 | 9.0 | 6 | 0 |
| merge_intervals | 2 | yes | yes | no | 7 | 28687 | $0.0081 | 9.9 | 6 | 0 |
| merge_intervals | 3 | yes | yes | no | 6 | 18753 | $0.0054 | 8.8 | 6 | 0 |
| parse_quoted_csv | 1 | yes | yes | no | 8 | 27051 | $0.0077 | 11.3 | 8 | 0 |
| parse_quoted_csv | 2 | yes | no | no | 10 | 41054 | $0.0118 | 18.8 | 11 | 0 |
| parse_quoted_csv | 3 | yes | no | no | 10 | 38736 | $0.0111 | 18.7 | 11 | 0 |
| lru_get_recency | 1 | yes | no | no | 10 | 37725 | $0.0108 | 17.7 | 11 | 0 |
| lru_get_recency | 2 | yes | yes | no | 6 | 19849 | $0.0057 | 9.7 | 7 | 0 |
| lru_get_recency | 3 | yes | yes | no | 7 | 23674 | $0.0068 | 11.5 | 8 | 0 |
| exponential_retry_delay | 1 | yes | yes | no | 5 | 15481 | $0.0044 | 5.9 | 5 | 0 |
| exponential_retry_delay | 2 | yes | no | no | 10 | 38214 | $0.0109 | 18.4 | 11 | 0 |
| exponential_retry_delay | 3 | yes | no | no | 10 | 36745 | $0.0105 | 17.8 | 11 | 0 |
| greedy_word_wrap | 1 | yes | yes | no | 5 | 15393 | $0.0044 | 6.1 | 6 | 0 |
| greedy_word_wrap | 2 | yes | yes | no | 6 | 18172 | $0.0052 | 7.5 | 5 | 0 |
| greedy_word_wrap | 3 | yes | no | no | 10 | 52445 | $0.0150 | 21.1 | 11 | 0 |
| normalize_relative_path | 1 | no | no | no | 10 | 39488 | $0.0113 | 20.1 | 12 | 0 |
| normalize_relative_path | 2 | no | no | no | 10 | 38401 | $0.0110 | 16.6 | 12 | 1 |
| normalize_relative_path | 3 | no | no | no | 10 | 38831 | $0.0111 | 17.6 | 11 | 1 |
| top_k_descending | 1 | yes | no | no | 10 | 36893 | $0.0105 | 17.9 | 12 | 0 |
| top_k_descending | 2 | yes | yes | no | 6 | 19065 | $0.0054 | 7.2 | 7 | 0 |
| top_k_descending | 3 | yes | yes | no | 5 | 15425 | $0.0044 | 6.7 | 6 | 0 |
| chunked_tail | 1 | yes | yes | no | 5 | 15939 | $0.0046 | 8.2 | 6 | 0 |
| chunked_tail | 2 | yes | yes | no | 7 | 22167 | $0.0063 | 10.5 | 7 | 0 |
| chunked_tail | 3 | yes | yes | no | 6 | 19719 | $0.0057 | 8.6 | 7 | 0 |
| binary_search_index | 1 | yes | no | no | 10 | 39204 | $0.0112 | 18.2 | 11 | 0 |
| binary_search_index | 2 | yes | yes | no | 6 | 19124 | $0.0055 | 6.9 | 5 | 0 |
| binary_search_index | 3 | yes | yes | no | 10 | 39050 | $0.0112 | 19.0 | 11 | 0 |
| recursive_flatten | 1 | yes | yes | no | 7 | 20711 | $0.0059 | 9.0 | 6 | 0 |
| recursive_flatten | 2 | yes | yes | no | 6 | 17514 | $0.0050 | 6.9 | 5 | 0 |
| recursive_flatten | 3 | yes | yes | no | 5 | 15021 | $0.0043 | 7.0 | 5 | 0 |
| config_port_validation | 1 | yes | no | no | 10 | 34472 | $0.0098 | 14.7 | 10 | 0 |
| config_port_validation | 2 | yes | no | no | 10 | 36801 | $0.0105 | 16.7 | 12 | 0 |
| config_port_validation | 3 | yes | yes | no | 9 | 33572 | $0.0096 | 15.1 | 11 | 0 |
| event_bus_snapshot | 1 | yes | no | no | 10 | 36383 | $0.0104 | 16.0 | 10 | 0 |
| event_bus_snapshot | 2 | yes | yes | no | 5 | 16126 | $0.0046 | 6.4 | 6 | 0 |
| event_bus_snapshot | 3 | yes | no | no | 10 | 38153 | $0.0109 | 16.7 | 10 | 0 |
| retry_policy_multi_file | 1 | no | no | no | 10 | 42131 | $0.0120 | 21.4 | 19 | 0 |
| retry_policy_multi_file | 2 | no | yes | yes | 8 | 29014 | $0.0083 | 14.9 | 12 | 0 |
| retry_policy_multi_file | 3 | yes | yes | no | 6 | 21543 | $0.0062 | 9.7 | 8 | 0 |
| rate_limiter_boundary | 1 | yes | no | no | 10 | 35616 | $0.0102 | 15.1 | 12 | 0 |
| rate_limiter_boundary | 2 | yes | yes | no | 10 | 40134 | $0.0115 | 20.4 | 12 | 0 |
| rate_limiter_boundary | 3 | yes | no | no | 10 | 38821 | $0.0111 | 18.2 | 14 | 0 |
| inventory_cancel_restock | 1 | yes | no | no | 10 | 37647 | $0.0107 | 16.9 | 12 | 0 |
| inventory_cancel_restock | 2 | yes | yes | no | 7 | 23784 | $0.0068 | 9.0 | 8 | 0 |
| inventory_cancel_restock | 3 | yes | yes | no | 6 | 19627 | $0.0056 | 8.7 | 7 | 0 |
| dependency_cycle_detection | 1 | yes | no | no | 10 | 37772 | $0.0108 | 16.8 | 12 | 0 |
| dependency_cycle_detection | 2 | yes | yes | no | 8 | 29207 | $0.0084 | 13.6 | 10 | 0 |
| dependency_cycle_detection | 3 | yes | yes | no | 10 | 39827 | $0.0114 | 19.2 | 11 | 0 |
| safe_path_resolution | 1 | yes | no | no | 10 | 42377 | $0.0122 | 23.7 | 13 | 0 |
| safe_path_resolution | 2 | yes | yes | no | 6 | 24333 | $0.0071 | 14.6 | 8 | 0 |
| safe_path_resolution | 3 | yes | no | no | 10 | 44978 | $0.0129 | 24.1 | 13 | 0 |

## Category summary

| Category | Passed | Attempted | False completions |
|---|---:|---:|---:|
| algorithms | 14 | 15 | 1 |
| boundaries | 9 | 9 | 0 |
| data_structures | 6 | 6 | 0 |
| multi_file | 7 | 9 | 1 |
| parsing | 6 | 9 | 0 |
| reliability | 3 | 3 | 0 |
| safety | 3 | 3 | 0 |
| state | 6 | 6 | 0 |
