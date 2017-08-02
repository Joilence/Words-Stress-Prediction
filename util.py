import re
def line_parser(line):
    data = {}

    data['wd'] = line.split(':')[0]

    data['ph'] = line.split(':')[1]

    v = []
    for u in data['ph'].split(' '):
        if re.match('^[A-Za-z]+$', u):
            continue
        v.append(u)
    data['v'] = v

    pri_pos = -1
    for index, u in enumerate(v):
        if u.endswith('1'):
            pri_pos = index + 1
    data['pri_pos'] = pri_pos

    pri_rvpos = -1 if pri_pos == -1 else len(v) + 1 - pri_pos
    data['pri_rvpos'] = pri_rvpos

    sec_pos = -1
    for index, u in enumerate(v):
        if u.endswith('2'):
            sec_pos = index + 1
    data['sec_pos'] = sec_pos

    sec_rvpos = -1 if sec_pos == -1 else len(v) + 1 - sec_pos
    data['sec_rvpos'] = sec_rvpos

    pos = sec_pos if pri_pos == -1 else pri_pos
    data['pos'] = pos

    rvpos = sec_rvpos if pri_rvpos == -1 else pri_rvpos
    data['rvpos'] = rvpos

    data['v_num'] = 0;
    for u in data['ph']:
        if re.match('[0-2]$', u):
            data['v_num'] += 1;
    return data

#test for remote branch