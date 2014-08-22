#!/usr/bin/python
import sys
import random

if __name__ == '__main__':
    if len(sys.argv) < 3:
	print 'usage: <input> <fold> [nfold=5]'
	exit(-1)

    if len( sys.argv ) > 3:
        nfold = int( sys.argv[3] )
    else:
        nfold = 5
    fold = int( sys.argv[2] )
    assert fold > 0 and fold  <= nfold  
    random.seed( 0 )
    
    fo = open( 'fold%d.txt'  % fold, 'w' )

    for l in open( sys.argv[1] ): 
        arr = l.split()
        uid,iid, sc = int(arr[0]),int(arr[1]), float(arr[2])
        if random.randint( 1, nfold ) == fold:
            # test is 1
            ngf = 1
        else: 
            ngf = 0
            
        fo.write('%f\t%d\t1\t1\t' % (sc, ngf ) )
        if ngf != 0:
            fo.write('0:0 ')
        fo.write('%d:1 %d:1\n' %(uid,iid))
    fo.close()

