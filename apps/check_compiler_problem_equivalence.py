import sys

def main():
    if len(sys.argv) < 3:
        print('Usage: python check_compiler_problem_equivalence.py filename1 filename2')
        return
    
    file1 = open(sys.argv[1]).read()
    file2 = open(sys.argv[2]).read()
    
    file1_lines = file1.split('\n')
    file2_lines = file2.split('\n')
    
    file1_intermediate_len_line = [line for line in file1_lines if line.startswith('f_log_intermediate_len = ')][0]
    file2_intermediate_len_line = [line for line in file2_lines if line.startswith('f_log_intermediate_len = ')][0]
    
    assert file1_intermediate_len_line == file2_intermediate_len_line
    
    file1_intermediate_subset_len_line = [line for line in file1_lines if line.startswith('f_log_intermediate_subset_len = ')][0]
    file2_intermediate_subset_len_line = [line for line in file2_lines if line.startswith('f_log_intermediate_subset_len = ')][0]
    
    assert file1_intermediate_subset_len_line == file2_intermediate_subset_len_line
    
    file1_const_len_line = [line for line in file1_lines if line.startswith('const_scale_bias_var_len = ')][0]
    file2_const_len_line = [line for line in file2_lines if line.startswith('const_scale_bias_var_len = ')][0]
    
    assert file1_const_len_line == file2_const_len_line
    
    file1_def_line = [line for line in file1_lines if line.startswith('def f')][0]
    file1_def_idx = file1_lines.index(file1_def_line)
    
    file2_def_line = [line for line in file2_lines if line.startswith('def f')][0]
    file2_def_idx = file2_lines.index(file2_def_line)
    
    count = 1
    
    while True:
        line1 = file1_lines[file1_def_idx + count].split('#')[0]
        line2 = file2_lines[file2_def_idx + count].split('#')[0]
        
        if line1 != line2:
            
            line1_frag = line1.replace('(', ' ').replace(')', ' ').replace(',', ' ').split(' ')
            line2_frag = line2.replace('(', ' ').replace(')', ' ').replace(',', ' ').split(' ')
            
            success = True
            
            if len(line1_frag) != len(line2_frag):
                success = False
            else:
                for i in range(len(line1_frag)):
                    frag1 = line1_frag[i]
                    frag2 = line2_frag[i]
                    
                    if frag1.startswith('var'):
                        frag1 = frag1.split('_')[0]
                        frag2 = frag2.split('_')[0]
                        
                    if frag1 != frag2:
                        success = False
                        break
                        
            if not success:
                print('Assertion fails at %s:%d and %s:%d' % (sys.argv[1], file1_def_idx + count + 1, sys.argv[2], file2_def_idx + count + 1))
                raise
            
        if 'return 0.0' in line1:
            break
            
        count += 1
        
    print('success')
    
if __name__ == '__main__':
    main()
        
    
