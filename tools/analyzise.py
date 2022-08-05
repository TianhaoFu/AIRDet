import json
import pprint
import argparse


def make_parser():
    parser = argparse.ArgumentParser("TRT profiler analyziser")
    parser.add_argument("--profile_path", default='airdet', type=str)
    return parser


args = make_parser().parse_args()
profile_path = args.profile_path

key_list = ['conv', 'copy', 'slice', 'resize', 'expand', 'gather', 'tile', 'pwn', 'transpose', 'pool', 'shuffle', 'reshape', 'softmax']

profile = json.load(open(profile_path,'rb'))


op_dict = {'others': 0., }
total_perc = 0
total_time = 0

others = 0
conv_num = 0
for detail in profile:
    if 'name' not in detail:
        continue
    name = detail['name']
    time = detail['averageMs']
    perc = detail['percentage']
    retv_flag = False
    for k in key_list:
        if k in name.lower():
            retv_flag = True
            if k not in op_dict:
                op_dict[k] = [time, 1]
            else:
                op_dict[k][0] += time
                op_dict[k][1] += 1
            if k == 'conv':
                conv_num += 1
            total_perc += perc
            total_time += time
            break
    if retv_flag == False:
        print(name)
        op_dict['others'] += time

total_time += op_dict['others']

pprint.pprint(op_dict)
print(f"total time: {total_time:.4f}ms")
print(f"total percent {total_perc:.4f}%")
print(f"num of conv: {conv_num}")


