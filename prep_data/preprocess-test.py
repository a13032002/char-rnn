
import sys
import pickle
from collections import defaultdict
import random

def output_sessions(sessions, f):
    for session in sessions:
        print >> f, ' '.join(session)

def process(input_file='../data/yoochoose/rsc15_test.txt', output_file='../data/yoochoose/yoochoose-sessions-test.dat', 
            dict_file='../data/yoochoose/item_dict.pkl', start_time=20140101000000000L, end_time=20141001000000000L, min_freq=2):
    '''
    Process the input data line by line, group data by session number (first attr), and filter by specified
    time stamp & minimum frequency.
    '''

    cur_session_items = []
    cur_session = "-1"
    cur_session_len = 0
    with open(dict_file) as f:
        item_dict = pickle.load(f)
    skip = False
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            if not skip:
                skip = True
                continue
            
            session, item, time = line.split('\t')[:3]
            time_n = long(time[:-3])
            #if time_n < start_time or time_n > end_time:
            #    continue

            if session == cur_session:
                cur_session_items.append(item)
                cur_session_len += 1
            else:
                if cur_session_len >= min_freq:
                    has_unseen_item = False
                    for item in cur_session_items:
                        if item not in item_dict:
                            has_unseen_item = True

                    if not has_unseen_item:
                        output_sessions([[item_dict[item] for item in cur_session_items]], f_out)

                cur_session = session
                cur_session_items = [item]
                cur_session_len = 1



def parse_time(time_string):
    date, time = time_string.split('T')
    year, month, day = date.split('-')
    hour, minute, second = time.split(':')
    second, msecond = second.split('.')
    return long(year + month + day + hour + minute + second + msecond)

if __name__ == '__main__':
    process() # TODO specify the arguments here or use the default values
    
