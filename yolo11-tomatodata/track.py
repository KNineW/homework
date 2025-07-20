import motmetrics as mm  # 导入该库
import numpy as np

metrics = list(mm.metrics.motchallenge_metrics)  # 即支持的所有metrics的名字列表
"""
['idf1', 'idp', 'idr', 'recall', 'precision', 'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_transfer', 'num_ascend', 'num_migrate']
"""

acc = mm.MOTAccumulator(auto_id=True)  #创建accumulator

# 用第一帧填充该accumulator
acc.update(
    [1, 2],                     # Ground truth objects in this frame
    [1, 2, 3],                  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],     # Distances from object 1 to hypotheses 1, 2, 3
        [0.5,  0.2,   0.3]      # Distances from object 2 to hypotheses 1, 2, 3
    ]
)

# 查看该帧的事件
print(acc.events) # a pandas DataFrame containing all events
# """