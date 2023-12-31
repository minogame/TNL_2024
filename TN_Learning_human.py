import numpy as np

np.random.seed(12)

order=4
Iteration=150000000
SGD_step=0.0000005
### Input data (in CP format)


dims=3
G1=np.random.randn(2,dims,2)
G2=np.random.randn(2,dims,2)
G3=np.random.randn(2,dims,2)
G4=np.random.randn(2,dims,2)
TR=np.tensordot(np.tensordot(np.tensordot(G1,G2,([2],[0])),G3,([3],[0])),G4,([4,0],[0,2]))


Y_label=TR.reshape(-1,order='F')
tnp=np.arange(dims**order).reshape((dims,dims,dims,dims),order='F')
X_feature=[]
for i in range(Y_label.shape[0]):
    tt=np.zeros([dims,order])
    tt[np.where(tnp==i)[0],0]=1
    tt[np.where(tnp==i)[1],1]=1
    tt[np.where(tnp==i)[2],2]=1
    tt[np.where(tnp==i)[3],3]=1
    X_feature.append(tt)



### Initialize the TensorNetwork Given the adjcency matrix 

TN_ranks=2
permutation=np.arange(order)
# np.random.shuffle(permutation)
adjacency_matrix=np.zeros([order,order],dtype=int)
shape_data=np.diag(np.ones(order,dtype=int)*dims)

for i in range(order-1):
    adjacency_matrix[i,i+1]=TN_ranks
    
adjacency_matrix[0,order-1]=TN_ranks

adjacency_matrix[np.tril_indices(order, -1)] = adjacency_matrix.transpose()[np.tril_indices(order, -1)]

permute=np.copy(permutation)

permutation_matrix=np.zeros([order,order],dtype=int)
for i in range(order):
				permutation_matrix[permute[i],i] = 1

adjacency_matrix=shape_data+np.matmul(np.matmul(permutation_matrix,adjacency_matrix),permutation_matrix.transpose())

adjacency_matrix[adjacency_matrix==0] = 1


TN_cores=[]

for i in range(order):
    temp=np.random.randn(*tuple(adjacency_matrix[i,:]))
    # temp=TenUnfold(temp,i)
    # for j in range(temp.shape[1]):
    #     temp[:,j]=temp[:,j]/np.linalg.norm(temp[:,j])
    
    # temp=Tenfold(temp,i,adjacency_matrix[i,:])
    TN_cores.append(temp)


# G1=np.random.randn(2,5,2)*5
# G2=np.random.randn(2,5,2)*5
# G3=np.random.randn(2,5,2)*5
# G1[0,:,:]=-G1[1,:,:]
# G3[:,:,0]=G3[:,:,1]
# TN_cores=[]
# TN_cores.append(G1.transpose((1,2,0)))
# TN_cores.append(G2)
# TN_cores.append(G3.transpose((2,0,1)))




#### generating contraction indedx
contration_index_mat=np.zeros([order,order],dtype=int) 
contration_index_mat[np.triu_indices(order, 1)]=np.arange(np.sum(np.arange(order-1)+1))+1
contration_index_mat=contration_index_mat+contration_index_mat.transpose()+np.diag(-1*(np.arange(order)+1))

# contration_index=[]
# for i in range(order):
#     contration_index.append(contration_index_mat[i,:].tolist())

contration_index=[]
for i in range(order):
    contration_index.append(contration_index_mat[i,:][adjacency_matrix[i,:]>1].tolist())
    TN_cores[i]=TN_cores[i].squeeze()


TN_cores_init=TN_cores.copy()

contraction_order=permutation.tolist()

Fully_contracting_index=Generate_FCing_index(contration_index,contraction_order)

Subchain_contracting_index=Generate_SCing_index(contration_index,contraction_order)


#### Initial loss
loss_tnp=0
for i in range(Y_label.shape[0]):
    input_tnp=[]
    for j in range(order):
        input_tnp.append(X_feature[i][:,j].copy().reshape(1,dims))
    loss_tnp=loss_tnp+(Y_label[i]-Full_ContractionCP(input_tnp,TN_cores,adjacency_matrix,contraction_order,Fully_contracting_index,contration_index))**2

print(np.sqrt(loss_tnp/np.inner(Y_label,Y_label)))


for itera in range(Iteration):
    #### gradient computing
    TN_cores_grads_sum=[]
    for i in range(order): 
        TN_cores_grads_sum.append(np.zeros(TN_cores[i].shape))
    for sample_index in range(Y_label.shape[0]):
        TN_cores_grads=[]
        for i in range(order): 
            input_tnp=[]
            for j in range(order):
                input_tnp.append(X_feature[sample_index][:,j].copy().reshape(1,dims))
            temp_matrix=SubchainCP(input_tnp,TN_cores,i,adjacency_matrix,contraction_order,Subchain_contracting_index,contration_index).reshape(-1,order='F')
            TN_cores_grads.append(Tenfold(2*np.outer(X_feature[sample_index][:,i],X_feature[sample_index][:,i]).dot(TenUnfold(TN_cores[i],np.where(np.array(contration_index[i])<0)[0].tolist()[0])).dot(np.outer(temp_matrix,temp_matrix))-2*Y_label[sample_index]*np.outer(X_feature[sample_index][:,i],temp_matrix),np.where(np.array(contration_index[i])<0)[0].tolist()[0],np.array(TN_cores[i].shape)))    
        for i in range(order): 
            TN_cores_grads_sum[i]=TN_cores_grads_sum[i]+TN_cores_grads[i]
    #### gradient desent
    for i in range(order):
        TN_cores[i]=TN_cores[i]+(-1)*SGD_step*TN_cores_grads_sum[i]


    loss_tnp=0
    for i in range(Y_label.shape[0]):
        input_tnp=[]
        for j in range(order):
            input_tnp.append(X_feature[i][:,j].copy().reshape(1,dims))
        loss_tnp=loss_tnp+(Y_label[i]-Full_ContractionCP(input_tnp,TN_cores,adjacency_matrix,contraction_order,Fully_contracting_index,contration_index))**2


    print('RSE:{}'.format(np.sqrt(loss_tnp/np.inner(Y_label,Y_label))))
    tnp_sum=0
    tnp_sum2=0
    for order_index in range(order):
        tnp_sum=tnp_sum+np.linalg.norm(TN_cores_init[order_index]-TN_cores[order_index])**2
        tnp_sum2=tnp_sum2+np.linalg.norm(TN_cores_init[order_index])**2
    print('change:{}'.format(np.sqrt(tnp_sum/tnp_sum2)))

