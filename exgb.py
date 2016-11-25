import numpy as np
import xgboost as xgb 
import sklearn.metrics

Ndata          = 5;
dataSets       = ['exgb1.csv','exgb2.csv','exgb3.csv','exgb4.csv','exgb5.csv'];
resutSets      = ['CV1year.csv','CV2year.csv','CV3year.csv','CV4year.csv','CV5year.csv'];
fillingMissing = -99999;

param = {'bst:eta':0.02, 
         'silent':1, 
         'objective':'binary:logistic',
         'bst:max_depth':11,
         'base_score':0.5,
         'subsample':0.9,
         'bst:colsample_bytree':0.5,
         'scale_pos_weight':1,
         'min_child_weight':5}
plst = param.items()
plst +=  [('eval_metric', 'auc')] 
num_round = 200
for l in range(0,Ndata):
    data = np.genfromtxt(dataSets[l], delimiter = ',')
    data[np.isnan(data)] = fillingMissing;
        
    data[np.isnan(data)] = 0;
    N = np.shape(data)[0];
    
    Nrep      = 10;
    Nop       = 4;
    NNewf     = 60;
    Threshold = 10;
    np.set_printoptions(precision = 4)
    
    numCV = 10;     
    resultsGether = np.zeros((numCV,Nrep+2),dtype = float);
    
    cvId =  np.random.randint(numCV, size = N);
    for i in range(0,numCV):
        num_features = np.shape(data)[1];
        dataTrain    = data[cvId!= i,:];
        dataTest     = data[cvId == i,:];
        labelsTrain  = dataTrain[:,(num_features-1)];
        dataTrain    = dataTrain[:,0:num_features-1];
    
        labelsTest         = dataTest[:,(num_features-1)];
        dataTest           = dataTest[:,0:num_features-1];
        NTrain             = np.shape(dataTrain)[0];
        NTest              = np.shape(dataTest)[0];
        num_features       = np.shape(dataTrain)[1];
        dtrain             = xgb.DMatrix( dataTrain, label = labelsTrain, missing = fillingMissing)
        dtest              = xgb.DMatrix( dataTest, label = labelsTest, missing = fillingMissing)
        evallist           = [(dtest,'eval'), (dtrain,'train')]
        bst                = xgb.train( plst, dtrain, num_round, evallist,early_stopping_rounds = 100)
        resultsGether[i,0] = float(bst.eval(dtest, 'test',0).split(":")[1]);
        p                  = np.zeros((num_features),dtype = float)
        #featuresNew       = np.zeros((NNewf,3),dtype = float)
        p                  = p  + 1/float(num_features);
        p_test             = np.zeros((NTest,Nrep),dtype = float);
        for j in range(0,Nrep):
            #here draw
            num_features = np.shape(p)[0];
            print np.shape(p)
            for k in range(0,NNewf):
                typeOpr = np.random.randint(Nop, size = 1);
                '''f1 = np.zeros((num_features,1),dtype = int);
                f2 = np.zeros((num_features,1),dtype = int);'''
                f1 = np.random.multinomial(1, p, size = 1)[0]
                f2 = np.random.multinomial(1, p, size = 1)[0]
                '''print('type: ' + str(typeOpr))
                print('f1: ' + str(f1.ravel().nonzero()))
                print('f2: ' + str(f2.ravel().nonzero()))'''
                '''featuresNew[k,0] = typeOpr;
                featuresNew[k,1] = np.where(f1!= 0)[0][0];
                featuresNew[k,2] = np.where(f2!= 0)[0][0]; '''
                if(typeOpr == 0):
                    newColTrain = dataTrain[:, f1 == 1] +dataTrain[:,f2 == 1];  
                    newColTest = dataTest[:, f1 == 1] +dataTest[:,f2 == 1]; 
                elif(typeOpr == 1):
                    newColTrain = dataTrain[:, f1 == 1] - dataTrain[:,f2 == 1];  
                    newColTest = dataTest[:, f1 == 1] - dataTest[:,f2 == 1];
                elif(typeOpr == 2): 
                    newColTrain = dataTrain[:, f1 == 1] * dataTrain[:,f2 == 1];  
                    newColTest = dataTest[:, f1 == 1] * dataTest[:,f2 == 1];
                else:
                    newColTrain = dataTrain[:, f1 == 1] / dataTrain[:,f2 == 1];  
                    newColTest = dataTest[:, f1 == 1] / dataTest[:,f2 == 1]; 
                dataTrain = np.append(dataTrain,newColTrain, axis = 1);
                dataTest = np.append(dataTest,newColTest, axis = 1);
                p = np.append(p,[0], axis = 0);
           # np.savetxt("featuresNew.csv", featuresNew, delimiter = ",")
            print np.shape(dataTrain)
            print np.shape(dataTest)
            dataTrain[np.isnan(dataTrain)] = 0;
            dataTest[np.isnan(dataTest)] = 0;
            dtrain               = xgb.DMatrix( dataTrain, label = labelsTrain, missing = fillingMissing)
            dtest                = xgb.DMatrix( dataTest, label = labelsTest, missing = fillingMissing)
            evallist             = [(dtest,'eval'), (dtrain,'train')]
            bst                  = xgb.train( plst, dtrain, num_round, evallist,early_stopping_rounds = 100)
            resultsGether[i,j+1] = float(bst.eval(dtest, 'test',0).split(":")[1]);
            p_test[:,j]          = bst.predict(dtest);
            fscore               = bst.get_fscore(fmap = '');
            if(j == Nrep-1):
                p_mean = np.mean(p_test, axis = 1);
                auc = sklearn.metrics.roc_auc_score(labelsTest, p_mean);
                resultsGether[i,j+2] = auc;
            print(fscore)
            keys = fscore.keys()
            print(len(keys))
            newColTrain = np.zeros((NTrain,1),dtype = float);
            newColTest = np.zeros((NTest,1),dtype = float);
            
            key = int(keys[0][1:]);
            p = np.zeros((len(keys)),dtype = float)+1/float(len(keys));
            newColTrain[:,0]  = dataTrain[:,key];
            newColTest[:,0]  = dataTest[:,key];
            dataTrainNEW = newColTrain;
            dataTestNEW = newColTest;
            for m in range(1,len(keys)):
                key              = int(keys[m][1:]);
                newColTrain[:,0] = dataTrain[:,key];
                newColTest[:,0]  = dataTest[:,key];
                dataTrainNEW     = np.append(dataTrainNEW,newColTrain, axis = 1);
                dataTestNEW      = np.append(dataTestNEW,newColTest, axis = 1);
                if(float(fscore[keys[m]])>Threshold):
                    p[m] = float(fscore[keys[m]]);
                else:
                    p[m] = 0;
            dataTrain = dataTrainNEW;
            dataTest = dataTestNEW;
            p = p/np.sum(p); 
    
    np.savetxt(resutSets[l], resultsGether, delimiter = ",")