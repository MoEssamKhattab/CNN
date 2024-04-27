from torch import nn
class CNNBlock(nn.Module):
    def __intit__(self, cin, cout, stride=1, groups=1):
        super().__init__()
        self.downsample = False

        self.cnn = nn.Conv2d(cin,cout,3,padding=1,stride=stride,bias=False, groups=groups)
        self.cnn2 = nn.Conv2d(cout,cout,3,padding=1,bias=False, groups=groups)
        if stride !=1:
            self.projection  = nn.Conv2d(cin,cout,1,stride=stride,bias=False)
            self.downsample = True
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.BN1 = nn.BatchNorm2d(cout)
        self.BN2 = nn.BatchNorm2d(cout)
    def forward(self,x):
        out = self.act1(self.BN1(self.cnn(x)))
        if self.downsample: return self.act2(self.BN2(self.cnn2(out)) + self.projection(x))
        return self.act2(self.BN2(self.cnn2(out)) + x)
