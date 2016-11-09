import sys
import pickle

def process(input_file='../data/yoochoose/yoochoose-clicks.dat', output_file='../data/yoochoose/yoochoose-sessions.dat', 
            dict_file='../data/yoochoose/item_dict.pkl', start_time=20140101000000000L, end_time=20141201000000000L, min_freq=2):
    '''
    Process the input data line by line, group data by session number (first attr), and filter by specified
    time stamp & minimum frequency.
    '''

    buf_size = 1000000
    buf = []
    cur_session_items = ""
    cur_session = "-1"
    cur_session_len = 0
    item_dict = {}
    num_items = 0
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            session, time, item = line.split(',')[:3]
            time_n = parse_time(time[:-1])
            if time_n < start_time or time_n > end_time:
                print "out"
                continue
            if item not in item_dict:
                num_items += 1
                item_dict[item] = str(num_items)
            item = item_dict[item]
            if session == cur_session:
                cur_session_items += " "
                cur_session_items += item
                cur_session_len += 1
            else:
                cur_session = session
                if cur_session_len >= min_freq:
                    cur_session_items += "\n"
                    buf.append(cur_session_items)
                cur_session_items = item
                cur_session_len = 1
                if len(buf) >= buf_size:
                    f_out.writelines(buf)
                    buf = []
        if cur_session_len >= min_freq:
            buf.append(cur_session_items)
        if len(buf) > 0:
            f_out.writelines(buf)

    with open(dict_file, 'wb') as dict_out:
        pickle.dump(item_dict, dict_out)

def parse_time(time_string):
    date, time = time_string.split('T')
    year, month, day = date.split('-')
    hour, minute, second = time.split(':')
    second, msecond = second.split('.')
    return long(year + month + day + hour + minute + second + msecond)

if __name__ == '__main__':
    process(input_file='../data/yoochoose/yoochoose-buys.dat') # TODO specify the arguments here or use the default values
    
