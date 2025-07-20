import motmetrics as mm  # 导入该库
import numpy as np
# 每行的第一个数是帧号；第二个数是id；后面接着的四个数表示框的位置和大小；然后倒数第四个数是置信度，
# 我这里设置为-1是因为我的跟踪方法根本不输出置信度，如果是你用的跟踪方法输出置信度，则你需要把这个数设为置信度；最后三个数不重要不用管。
gt_file="gt.txt"
ts_file="test.txt"


gt = mm.io.loadtxt(gt_file, fmt="mot15-2D")  # 读入GT
ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")  # 读入自己生成的跟踪结果

acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)  # 根据GT和自己的结果，生成accumulator，distth是距离阈值
# print(acc)

mh = mm.metrics.create()

# 打印单个accumulator
summary = mh.compute(acc,
                     metrics=['num_frames', 'mota', 'motp'], # 一个list，里面装的是想打印的一些度量
                     name='acc') # 起个名
print(summary)
"""
     num_frames  mota  motp
acc           3   0.5  0.34
"""

# 打印多个accumulators
summary = mh.compute_many([acc, acc.events.loc[0:1]], # 多个accumulators组成的list
                          metrics=['num_frames', 'mota', 'motp'],
                          names=['full', 'part']) # 起个名
print(summary)

strsummary = mm.io.render_summary(
    summary,
    formatters={'mota' : '{:.2%}'.format},  # 将MOTA的格式改为百分数显示
    namemap={'mota': 'MOTA', 'motp' : 'MOTP'}  # 将列名改为大写
)
print(strsummary)
"""
      num_frames   MOTA      MOTP
full           3 50.00%  0.340000
part           2 50.00%  0.166667
"""

# mh模块中有内置的显示格式
summary = mh.compute_many([acc, acc.events.loc[0:1]],
                          metrics=mm.metrics.motchallenge_metrics,
                          names=['full', 'part'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)