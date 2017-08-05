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

    data['vr'] = round(len(data['v'])/len(data['ph']), 2)

    return data

def conditional_res_test(arr, condition):
    report = ''
    stress_rv = [0, 0, 0, 0, 0, 0]
    stress = [0, 0, 0, 0, 0, 0]
    for s in arr:
        v_rvpos = line_parser(s)['rvpos']
        v_pos = line_parser(s)['pos']
        stress_rv[v_rvpos] += 1
        stress[v_pos] += 1
    report += '## ' + condition + ' ##' + '\n'
    report += 'stress: ' + str(stress) + '\n top_rate = ' + str(round(max(stress)/sum(stress),2)) + '\n'
    report += 'stress_rv: ' + str(stress_rv) + '\n top_rate_rv = ' + str(round(max(stress_rv)/sum(stress_rv),2)) + '\n'
    report += '---------' + '\n'

    return report
