
import rcca
import numpy as np

def align_cca(train,test,K=None,reg=0.1):
    X_train,Y_train=train[:,0,:],train[:,1,:]
    X_test,Y_test=test[:,0,:],test[:,1,:]

    cca = rcca.CCA(reg =reg, numCC=K,kernelcca=False,verbose=False)
    cca.train([X_train, Y_train])

    zx = np.dot(X_test,cca.ws[0])
    zy = np.dot(Y_test,cca.ws[1])

    return zx,zy