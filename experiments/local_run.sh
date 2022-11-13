python slow_task_switch.py --run-name slow_switch_ --datasets AffineCorrelatedGaussian --save-path "../results/task_switch_less_iters_local/" --change-task-every 1500
python slow_task_switch.py --run-name slow_switch_ --datasets MultiDimGaussian --save-path "../results/task_switch_less_iters_local/" --change-task-every 1500
python slow_task_switch.py --run-name mid_switch_ --datasets AffineCorrelatedGaussian --save-path "../results/task_switch_less_iters_local/" --change-task-every 600
python slow_task_switch.py --run-name mid_switch_ --datasets MultiDimGaussian --save-path "../results/task_switch_less_iters_local/" --change-task-every 600
