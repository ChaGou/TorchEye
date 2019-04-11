import torch
import math
class Camera:
    s = torch.Tensor([[640],[480]])
    c = s / 2
    f = torch.Tensor([[766], [766]])
    def getAlBefromPixal(self, pixalX, pixalY):
        x1 = (pixalX - self.c[0, 0]) / self.f[0, 0]
        y1 = (pixalY - self.c[1, 0]) / self.f[1, 0]
        temp =  torch.sqrt(x1 * x1 + y1 * y1)
        al = torch.atan2(y1, x1)
        be = torch.atan(1/temp)
        al[temp <= 0.001] = 0
        be[temp <= 0.001] = math.pi / 2
        return al, be

    def getPixalFromAlBe(self,al, be):
        x1 = math.cos(al)/math.tan(be)
        y1 = math.sin(al)/math.tan(be)
        return torch.Tensor([x1*self.f[0,0]+self.c[0,0],y1*self.f[1,0]+self.c[1,0]]).view(2,1)

    def getUniformPixalFromAlBe(self, al, be):
        x1 = math.cos(al) / math.tan(be)
        y1 = math.sin(al) / math.tan(be)
        return torch.Tensor([(x1 * self.f[0, 0] + self.c[0, 0])/self.s[0,0], (y1 * self.f[1, 0] + self.c[1, 0])/self.s[1,0]]).view(2, 1)

cam = Camera()
al,be=cam.getAlBefromPixal(torch.Tensor([253,267]),torch.Tensor([[269,284]]))
print(al/math.pi*180,be/math.pi*180,)

#  131.2810  106.9170
# [torch.cuda.FloatTensor of size 1x2 (GPU 0)]
#
#
#  158.0000  112.5000