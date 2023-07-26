import torch.nn as nn 
import torch

class dilate_conv(nn.Module):
    def __init__(self,inchanels, outchanels,dilate):
        super(dilate_conv,self).__init__()
        #a = np.zeros(dilate)
        #self.t = torch.from_numpy(a)
        self.conv = nn.Conv2d(inchanels, outchanels, kernel_size=(2,3),padding=(0,1),stride=1,dilation=(dilate,1))
        self.relu = nn.PReLU()
    def forward(self,x):
        #out = torch.cat((x,self.t),-1)
        out = self.relu(self.conv(x))
        return out
        
class wavenet2d(nn.Module):
    def __init__(self):
        '''
        opts: the system para
        '''
        super(wavenet2d,self).__init__()

        self.P = 11
        self.G0 = 64
        self.T = 2048
        self.trace = 3
        
        self.conv2 = nn.Conv2d(1,self.G0,kernel_size=3,padding = 1,stride= 1)
        convs = []
        for i in range(self.P):
            convs.append(dilate_conv(self.G0,self.G0,2**i))
        self.dlconv = nn.Sequential(*convs)
        self.SFE2 = nn.Conv2d(self.G0,1,kernel_size=3,padding = 1,stride= 1)
        self.fc = nn.Linear(self.T*self.trace, self.T*self.trace)
        
        #init
        for para in self.modules():
            if isinstance(para,nn.Conv1d):
                nn.init.orthogonal_(para.weight)
            if isinstance(para,nn.Conv2d):
                nn.init.orthogonal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()
            if isinstance(para,nn.Linear):
                nn.init.orthogonal_(para.weight)

    def forward(self,x):
        out = self.conv2(x)
        out = self.dlconv(out)
        out = self.SFE2(out)
        #out = self.SFE3(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        out = out.view(-1, 1, self.T, self.trace)
        
        return out