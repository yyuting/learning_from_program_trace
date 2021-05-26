import os
import sys
import numpy
import pickle

def main():
    file = sys.argv[1]
    loop_dict = analyze_loop_statistic(file)
    generate_collect_loop_statistic_code(loop_dict)

def generate_collect_loop_statistic_code(loop_dict):
    s = []
    for loop_name, loop_vals in loop_dict.items():
        for loop_var_name, loop_var_info in loop_vals.items():
            all_loop_var_names = []
            all_loop_var_square = []
            for i in range(loop_var_info['max_iter'] + 1):
                all_loop_var_names.append(loop_name + '_' + str(i) + '_' + loop_var_name)
                all_loop_var_square.append(all_loop_var_names[-1] + '*' + all_loop_var_names[-1])
            #line_sum = 'vec_output[%d] = (%s) / %d' % (loop_var_info['sum'], '+'.join(all_loop_var_names), loop_var_info['max_iter'] + 1)
            #line_square = 'vec_output[%d] = (%s) / %d - vec_output[%d]' % (loop_var_info['square'], '+'.join(all_loop_var_square), loop_var_info['max_iter'] + 1, loop_var_info['sum'])
            line_sum = 'vec_output_%d = (%s) / %d' % (loop_var_info['sum'], '+'.join(all_loop_var_names), loop_var_info['max_iter'] + 1)
            line_square = 'vec_output_%d = (%s) / %d - vec_output_%d * vec_output_%d' % (loop_var_info['square'], '+'.join(all_loop_var_square), loop_var_info['max_iter'] + 1, loop_var_info['sum'], loop_var_info['sum'])
            line_vec_output_sum = 'vec_output[%d] = vec_output_%d * %d' % (loop_var_info['sum'], loop_var_info['sum'], loop_var_info['max_iter'] + 1)
            line_vec_output_square = 'vec_output[%d] = vec_output_%d * %d' % (loop_var_info['square'], loop_var_info['square'], loop_var_info['max_iter'] + 1)
            s.append(line_sum)
            s.append(line_square)
            s.append(line_vec_output_sum)
            s.append(line_vec_output_square)
    s = '\n'.join(s)
    #open('manual_collect.py', 'w').write(s)
    return s

def analyze_loop_statistic(file):
    lines = open(file).read().split('\n')
    loop_dict = {}
    for line in lines:
        if 'loop sum' in line or 'loop square' in line:
            assert 'vec_output[' in line

            vec_idx = line.index('vec_output[')
            vec_idx_end = line.index(']')
            vec_output_order = int(line[vec_idx+len('vec_output['):vec_idx_end])

            indicator = 'sum' if 'loop sum' in line else 'square'
            idx = line.index('loop ' + indicator)
            info_line = line[idx + len('loop ' + indicator + ' for '):]
            loop_name_end_idx = info_line.index(',')
            loop_name = info_line[:loop_name_end_idx]

            info_line = info_line[loop_name_end_idx+2:]
            var_name_end_idx = info_line.index(',')
            var_name = info_line[:var_name_end_idx]

            info_line = info_line[var_name_end_idx+11:]
            max_iter_end_idx = info_line.index('#')
            max_iter = int(info_line[:max_iter_end_idx])

            if loop_name not in loop_dict.keys():
                loop_dict[loop_name] = {}
            if var_name not in loop_dict[loop_name]:
                loop_dict[loop_name][var_name] = {}
            assert indicator not in loop_dict[loop_name][var_name].keys()
            loop_dict[loop_name][var_name][indicator] = vec_output_order
            if 'max_iter' in loop_dict[loop_name][var_name].keys():
                assert max_iter == loop_dict[loop_name][var_name]['max_iter']
            else:
                loop_dict[loop_name][var_name]['max_iter'] = max_iter

    for loop_name, loop_vals in loop_dict.items():
        max_iter = None
        var_collection = []
        for loop_var_name, loop_var_info in loop_vals.items():
            var_collection.append(int(loop_var_name))
            if max_iter is not None:
                assert max_iter == loop_var_info['max_iter']
            else:
                max_iter = loop_var_info['max_iter']
            assert 'sum' in loop_var_info.keys()
            assert 'square' in loop_var_info.keys()
        max_var_count = numpy.max(var_collection)
        assert sorted(var_collection) == list(range(max_var_count + 1))

    #numpy.savez('loop_info.npz', **loop_dict)
    #pickle.dump(loop_dict, open('loop_info', 'wb'))
    print('success')
    return loop_dict

if __name__ == '__main__':
    main()
