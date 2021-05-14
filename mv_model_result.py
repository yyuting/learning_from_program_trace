import sys
import os
import shutil

"""
helper utilit: move a more compact trained model to permanent storage
"""

def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    
    if src.endswith('/'):
        src = src[:-1]
    
    _, dirname = os.path.split(src)
    
    files = os.listdir(src)
    
    assert not os.path.isdir(os.path.join(dst, dirname))
    
    os.mkdir(os.path.join(dst, dirname))
    
    for file in files:
        can_copy = True
        try:
            epoch = int(file)
            can_copy = False
        except:
            pass
        
        if file.startswith('model') or file.startswith('checkpoint'):
            if 'boids' not in dirname:
                can_copy = False
            
        if file == '__pycache__':
            can_copy = False
            
        if not can_copy:
            continue
            
        if os.path.isdir(os.path.join(src, file)):
            shutil.copytree(os.path.join(src, file), os.path.join(dst, dirname, file))
        else:
            shutil.copyfile(os.path.join(src, file), os.path.join(dst, dirname, file))
            
if __name__ == '__main__':
    main()