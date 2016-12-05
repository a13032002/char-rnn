import matplotlib.pyplot as plt
import numpy as np

def loadStat(path):
    f = open(path)
    rec = {}
    mrr = {}
    length_index = []

    for line in f:
        if len(line.strip()) == 0: continue
        tokens = line.strip().split("\t")
        l = int(tokens[0])
        length_index.append(l)
        if l in rec:
            rec[l].append(float(tokens[1]))
        else:
            rec[l] = [float(tokens[1])]
        if l in mrr:
            mrr[l].append(float(tokens[2]))
        else:
            mrr[l] = [float(tokens[2])]

    length_index = list(set(length_index))
    length_index.sort()
    return rec, mrr, length_index

def calAvg(stats):
    for k, val in stats.iteritems():
        stats[k] = np.average(val)
    return stats

def chunkAvg(stats, labels):
    new_stats = {}
    for l in labels:
        new_stats[l] = []
    for k,v in stats.iteritems():
        new_stats[labels[(k-1) / 10]] += v

    for k,v in new_stats.iteritems():
        new_stats[k] = np.average(v)
    return new_stats

if __name__ == "__main__":
    fnames = ['../result/gru-128', '../result/gru-256', '../result/gru-att-128', '../result/gru-att-256']
    labels = ['gru-128', 'gru-256', 'gru-att-128', 'gru-att-256']
    recs = []
    mrrs = []
    indices = []
    ranges = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']
    colors = ['c', 'm', 'g', 'r']

    for path in fnames:
        rec, mrr, length_index = loadStat(path)
        recs.append(chunkAvg(rec, ranges))
        mrrs.append(chunkAvg(mrr, ranges))
        indices.append(length_index)

    fig, ax = plt.subplots()
    for i in range(len(fnames)):
        print [recs[i][indx] for indx in ranges]
        ax.bar(np.linspace(0+i,40+i,9)[:-1], [recs[i][indx] for indx in ranges], color=colors[i], label=labels[i])
    ax.set_xticks(np.linspace(2,42,9)[:-1])
    ax.set_xticklabels(ranges)
    ax.legend(loc=2)
    plt.show()

    fig, ax = plt.subplots()
    for i in range(len(fnames)):
        print [recs[i][indx] for indx in ranges]
        ax.bar(np.linspace(0+i,40+i,9)[:-1], [mrrs[i][indx] for indx in ranges], color=colors[i], label=labels[i])
    ax.set_xticks(np.linspace(2,42,9)[:-1])
    ax.set_xticklabels(ranges)
    ax.legend(loc=2)
    plt.show()