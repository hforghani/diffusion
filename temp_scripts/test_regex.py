import re
from time import time
import settings


def read_network():
    i = 1
    network = {}
    with open(settings.WEIBO_FOLLOWERS_PATH, encoding='utf-8') as f:
        f.readline()
        line = f.readline()
        while line:
            parts = [int(n) for n in line.strip().split()]
            network[parts[0]] = {}
            n = parts[1]
            for j in range(n):
                network[parts[0]][parts[2 + 2 * j]] = parts[3 + 2 * j]
            if i % 100000 == 0:
                print('{} lines read'.format(i))
            line = f.readline()
            i += 1
    return network


def read_network_str():
    i = 1
    with open(settings.WEIBO_FOLLOWERS_PATH, encoding='utf-8') as f:
        content = f.read()
    return content


def predecessors(net, uid_i):
    results = set()
    if uid_i in net:
        results.update(set(net[uid_i].keys()))
    for u, followees in net.items():
        if uid_i in followees.keys() and followees[uid_i] == 1:
            results.add(u)
    return results


def successors_str(content, uid_i):
    succ = []
    res = re.findall(r'.*\s+{}\b.*'.format(uid_i), content, re.MULTILINE)
    for line in res:
        m = re.match(r'(\d+)\s', line)
        if m and m.groups():
            succ.append(m.groups()[0])
        else:
            print('Error: the line does not have a number at the beginning!')
    return succ


def predecessors_str(content, uid_i):
    res = re.search(r'^({}\s+.*)'.format(uid_i), content, re.MULTILINE)
    if res and res.groups():
        line = res.groups()[0]
        parts = [int(n) for n in line.strip().split()]
        n = parts[1]
        pred = parts[2:2 + 2 * n:2]
    else:
        print('INFO: no followees found for user index {}'.format(uid_i))
        pred = []
    return pred


uid_i = 8

t0 = time()
print('reading network ...')
#net = read_network()
content = read_network_str()
print('network read in %ds' % (time() - t0))

t0 = time()
print('extracting predecessors ...')
#pred = predecessors(net, uid_i)
pred = predecessors_str(content, uid_i)
print('predecessors done in %ds' % (time() - t0))
print('predecessors of {}: {}'.format(uid_i, pred))

#t0 = time()
#print('extracting successors ...')
##succ = successors(net, uid_i)
#succ = successors_str(content, uid_i)
#print('successors done in %ds' % (time() - t0))
#print('successors of {}: {}'.format(uid_i, succ))

