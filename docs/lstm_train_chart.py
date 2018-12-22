#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import re

LINE_REGEX = re.compile(r'\[TextLSTM\]\s+Epoch\s+(\d+)\s+\((\d+\.\d+)s\)\s+Average\sloss=(\d+\.\d+)\s+Dev\sF1=(\d+\.\d+)')

def parse_file(filename, line_regex):
    xs, elapsed_times, losses, dev_f1s = [], [], [], []
    fin = open(filename, "r")
    for line in fin:
        line = line.strip()
        m = re.match(line_regex, line)
        if m is not None:
            xs.append(int(m.group(1)))
            elapsed_times.append(float(m.group(2)))
            losses.append(float(m.group(3)))
            dev_f1s.append(float(m.group(4)))
    fin.close()
    return xs, elapsed_times, losses, dev_f1s

xs_c, ets_c, ls_c, dfs_c = parse_file("./cpu_lstm.txt", LINE_REGEX)
xs_g, ets_g, ls_g, dfs_g = parse_file("./gpu_lstm.txt", LINE_REGEX)

plt.figure(figsize=(5, 10))

plt.subplot(311)
plt.plot(xs_c, ets_c, color="blue", marker="o", label="CPU")
plt.plot(xs_c, ets_g, color="green", marker="o", label="GPU")
plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
plt.xlabel("epochs")
plt.ylabel("elapsed (s)")
plt.legend(loc="best")

plt.subplot(312)
plt.plot(xs_c, ls_c, color="blue", marker="o", label="CPU")
plt.plot(xs_c, ls_g, color="green", marker="o", label="GPU")
plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="best")

plt.subplot(313)
plt.plot(xs_c, dfs_c, color="blue", marker="o", label="CPU")
plt.plot(xs_c, dfs_g, color="green", marker="o", label="GPU")
plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
plt.xlabel("epochs")
plt.ylabel("Dev F1")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

