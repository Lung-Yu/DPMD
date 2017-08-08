import os
from Modules.Parsers import ObjdumpParser

def get_max_length(DATA_DIR):
    max_len = 0

    for fn in os.listdir(DATA_DIR):
        parser = ObjdumpParser()
        parser.loadfile(os.path.join(DATA_DIR,fn))
        vector = parser.asm2vec()
        clens = len(vector)
        if clens > max_len:
            max_len = clens
        print ("max %d ; %s -> len : %d")%(max_len,fn,clens)
        del parser,vector,clens
        
    return max_len

if __name__ == '__main__':
    max_len = get_max_length('../samples/Malware_Samples/ASM_Malekal')
    print ("finish : %d ")%(max_len)
    
        