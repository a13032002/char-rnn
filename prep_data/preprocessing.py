import sys
import pickle
from collections import defaultdict
import random

random.seed(0)
def output_sessions(sessions, f):
    random.shuffle(sessions)
    for session in sessions:
        print >> f, ' '.join(session)

def process(input_file='../data/yoochoose/yoochoose-clicks.dat', output_file='../data/yoochoose/yoochoose-sessions.dat', 
            dict_file='../data/yoochoose/item_dict.pkl', start_time=20140101000000000L, end_time=20140801000000000L, min_freq=2):
    '''
    Process the input data line by line, group data by session number (first attr), and filter by specified
    time stamp & minimum frequency.
    '''

    batch_size = 16
    sessions = defaultdict(list)
    cur_session_items = []
    cur_session = "-1"
    cur_session_len = 0
    item_dict = {}
    num_items = 0
    max_session_len = 0
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            session, time, item = line.split(',')[:3]
            time_n = parse_time(time[:-1])
            if time_n < start_time or time_n > end_time:
                continue
            if item not in item_dict:
                num_items += 1
                item_dict[item] = str(num_items)
            item = item_dict[item]
            if session == cur_session:
                cur_session_items.append(item)
                cur_session_len += 1
            else:
                max_session_len = max(max_session_len, cur_session_len)
                if cur_session_len >= min_freq:
                    sessions[cur_session_len].append(cur_session_items)
                    if len(sessions[cur_session_len]) == batch_size:
                        output_sessions(sessions[cur_session_len], f_out)
                        sessions[cur_session_len] = []

                cur_session = session
                cur_session_items = [item]
                cur_session_len = 1

    print len(item_dict), max_session_len
    with open(dict_file, 'wb') as dict_out:
        pickle.dump(item_dict, dict_out)

    lost = 0
    for k in sessions:
        lost += len(sessions[k])
    print lost
def parse_time(time_string):
    date, time = time_string.split('T')
    year, month, day = date.split('-')
    hour, minute, second = time.split(':')
    second, msecond = second.split('.')
    return long(year + month + day + hour + minute + second + msecond)

if __name__ == '__main__':
    process(input_file='../data/yoochoose/yoochoose-buys.dat') # TODO specify the arguments here or use the default values
    
