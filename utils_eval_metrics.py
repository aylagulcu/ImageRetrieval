import torch
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F


def create_feat_vecs(dataset, model, batch_size=400):
    """
    Create a feture vector for each image in the dataset
    returns a [M, embed] size matrix, where M is the number of images in the dataset and embed is the embedding size which is defined by the model 
    """

    Fvecs = []
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size, sampler=SequentialSampler(dataset))
    torch.set_grad_enabled(False)
    model.eval()
    for data, target in dataLoader:
        #inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
        #fvec = model(inputs_bt.cuda())
        
        # data.shape: [400, 3, 224, 224]; target.shape: [400] 
        
        if cuda:
            data = data.to(device)
        
        fvec = model.get_embedding(data) # fvec.shape: [400, 128]

        fvec = F.normalize(fvec, p = 2, dim = 1).cpu() 
        # F.normalize:  performs L2 normalization (p=2) over each row vector (dim=1) 
        # by first calculating the norm of each vector to find max norm; then divides each vector to this max norm.
        # Also known as Euclidean norm
        
        Fvecs.append(fvec)
        
    return torch.cat(Fvecs,0) # Concatenates the given sequence of seq tensors in the given dimension.


def create_sim_matrix(Fvec, imgLab):
    # imgLab numpy.ndarray like shape: (8041,)
    # Fvec shape is like [8041, 128], where the second dimension is the embedding size
   
    N = len(imgLab) #8041 labels

    imgLab = torch.LongTensor([imgLab[i] for i in range(len(imgLab))])
    # imgLab.shape: [8041]
    # Fvec.shape: [8041, 128]
    
    D = Fvec.mm(torch.t(Fvec)) # mm: matrix multiplication. (n×m) mm (m×p) results in  (n×p) tensor.
    # [8041, 128] mm [128, 8041] --> [8041, 8041] this is D matrix
    # There are 1's along the diagonal!
    
    D[torch.eye(len(imgLab)).bool()] = -1 
    # torch.eye: Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
    # D[torch.eye(len(imgLab)).bool()]: diagonal elements of D will take a value of -1 ; the rest will remain the same

    return D



def recall(D, imgLab,rank=None):
    """
        Given similarity matrix; calculates recall based on all images. 
        For each image, a True is counted for each retrieved image i, i=1..,k, if it has the same label

        # D similarity matrix is like[8041, 8041] 
        # imgLab numpy.ndarray like shape: (8041,)

    """

    N = len(imgLab) #8041 labels

    imgLab = torch.LongTensor([imgLab[i] for i in range(len(imgLab))])
    # imgLab.shape: [8041]
       
    if rank==None: # only rank 1 is computed
        _,idx = D.max(1) # returns both values and indices; dim=1 means returns for each row 
        imgPre = imgLab[idx]
        A = (imgPre==imgLab).float()
        return (torch.sum(A)/N).item()
    else:
        _,idx = D.topk(rank[-1])
        acc_list = []
        for r in rank:
            A = 0
            for i in range(r):
                imgPre = imgLab[idx[:,i]]
                A += (imgPre==imgLab).float()
            acc_list.append((torch.sum((A>0).float())/N).item())
        return torch.Tensor(acc_list)


