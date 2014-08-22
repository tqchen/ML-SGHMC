#!/usr/bin/python
import sys
import random

if __name__ == '__main__':
    if len(sys.argv) < 2:
	print 'usage: <input> <output>'
	exit(-1)

    umap = {}
    imap = {}
    
    fo = open( sys.argv[2], 'w' )
    for line in open( sys.argv[1] ):
        arr = line.split('::')
        uid = int( arr[0] )
        iid = int( arr[1] )
        if uid not in umap:
            umap[ uid ] = len( umap )
        if iid not in imap:
            imap[ iid ] = len( imap )
        score = int( arr[2] ) - 1
        fo.write( '%d\t%d\t%f\n' % (  umap[uid], imap[iid], score/ 4.0 ) )

    fo.close()
    
    print 'nuser=%d, nitem=%d' % ( len(umap), len(imap) )
    sc = [ float(l.split('::')[2]) - 1  for l in open( sys.argv[1]) ]      
    print 'score=[0-4], mean = %f' % ( sum(sc) / len(sc) / 4.0 )
