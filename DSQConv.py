import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundWithGradient(torch.autograd.Function):#输入为ψ函数
    @staticmethod
    def forward(ctx, x):
        delta = torch.max(x) - torch.min(x)
        x = (x/delta + 0.5)
        return x.round() * 2 - 1#这里就是模拟了将-1到1的数据round到0和1两个数
        #return
    @staticmethod
    def backward(ctx, g):
        return g 
#自定义通道合并的//和%不可导的导数
class Qactivation_split(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Qactivation):
        Qactivation_nbit_h = Qactivation // (2 ** 4)
        Qactivation_nbit_l = Qactivation % (2 ** 4)

        return Qactivation_nbit_h,Qactivation_nbit_l
    @staticmethod
    def backward(ctx, g_nbit_h,g_nbit_l):
        return g_nbit_h*(1/(2**5))+g_nbit_l*(1/2)


class DSQConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                momentum = 0.1,                
                num_bit = 8, QInput = True, bSetQ = True):#bSetQ = True
        super(DSQConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        #self.bit_range = 2**self.num_bit -1	 #计算比特数
        self.is_quan = bSetQ        
        self.momentum = momentum
        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization#32位整数
            # Weight
            self.uW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.lW  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
            self.register_buffer('running_uw', torch.tensor([self.uW.data])) # init with uw
            self.register_buffer('running_lw', torch.tensor([self.lW.data])) # init with lw
            self.alphaW = nn.Parameter(data = torch.tensor(0.2).float())
            # Bias
            if self.bias is not None:
                self.uB = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.lB  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
                self.register_buffer('running_uB', torch.tensor([self.uB.data]))# init with ub
                self.register_buffer('running_lB', torch.tensor([self.lB.data]))# init with lb
                self.alphaB = nn.Parameter(data = torch.tensor(0.2).float())
                

            # Activation input	#修改split
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.lA  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
                self.register_buffer('running_uA', torch.tensor([self.uA.data])) # init with uA
                self.register_buffer('running_lA', torch.tensor([self.lA.data])) # init with lA
                self.alphaA = nn.Parameter(data = torch.tensor(0.2).float())

    def clipping(self, x, upper, lower):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x

    def phi_function(self, x, mi, alpha, delta):#ψ函数，就是近似round函数

        # alpha should less than 2 or log will be None
        # alpha = alpha.clamp(None, 2)
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
        s = 1/(1-alpha)
        k = (2/alpha - 1).log() * (1/delta)
        x = (((x - mi) *k ).tanh()) * s 
        return x	

    def sgn(self, x):
        x = RoundWithGradient.apply(x)

        return x

    def dequantize(self, x, lower_bound, delta, interval):#文章中的Q函数，反量化的值

        # save mem
        x =  ((x+1)/2 + interval) * delta + lower_bound#(x+1)/2将范围限定在[0,1]

        return x



    def forward(self, x):

        bit_range=2**self.num_bit -1
        #print(bit_range)
        if self.is_quan:
            # Weight Part
            # moving average
            if self.training:
                cur_running_lw = self.running_lw.mul(1-self.momentum).add((self.momentum) * self.lW)
                cur_running_uw = self.running_uw.mul(1-self.momentum).add((self.momentum) * self.uW)
            else:
                cur_running_lw = self.running_lw
                cur_running_uw = self.running_uw

            Qweight = self.clipping(self.weight, cur_running_uw, cur_running_lw)
            cur_max = torch.max(Qweight)
            cur_min = torch.min(Qweight)
            delta =  (cur_max - cur_min)/(bit_range)
            interval = (Qweight - cur_min) //delta            
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.alphaW, delta)
            Qweight = self.sgn(Qweight)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval)

            Qbias = self.bias
            # Bias			
            if self.bias is not None:
                # self.running_lB.mul_(1-self.momentum).add_((self.momentum) * self.lB)
                # self.running_uB.mul_(1-self.momentum).add_((self.momentum) * self.uB)
                if self.training:
                    cur_running_lB = self.running_lB.mul(1-self.momentum).add((self.momentum) * self.lB)
                    cur_running_uB = self.running_uB.mul(1-self.momentum).add((self.momentum) * self.uB)
                else:
                    cur_running_lB = self.running_lB
                    cur_running_uB = self.running_uB

                Qbias = self.clipping(self.bias, cur_running_uB, cur_running_lB)
                cur_max = torch.max(Qbias)
                cur_min = torch.min(Qbias)
                delta =  (cur_max - cur_min)/(bit_range)
                interval = (Qbias - cur_min) //delta#在0-bit_range的哪一个bin中
                mi = (interval + 0.5) * delta + cur_min
                Qbias = self.phi_function(Qbias, mi, self.alphaB, delta)
                Qbias = self.sgn(Qbias)
                Qbias = self.dequantize(Qbias, cur_min, delta, interval)

            # Input(Activation)#修改split
            Qactivation = x#如果不量化激活值就直接将输入的激活值给Qactivation
            if self.quan_input:
                                
                if self.training:

                    cur_running_lA = self.running_lA.mul(1-self.momentum).add((self.momentum) * self.lA)
                    cur_running_uA = self.running_uA.mul(1-self.momentum).add((self.momentum) * self.uA)
                else:
                    cur_running_lA = self.running_lA
                    cur_running_uA = self.running_uA
                "*********************** 近似round函数，然后round(量化)*************************"
                #对于这一部分应该就是需要改的点，修改split，需要修改quantize之后dequantize之前的部分。把一个2nbit拆分成2个nbit
                Qactivation = self.clipping(x, cur_running_uA, cur_running_lA)
                cur_max = torch.max(Qactivation)
                cur_min = torch.min(Qactivation)
                #if self.flag_split: #flag_split由train.py会判定，如果需要拆分bit则flag_split=True
                    #Qweight_h=Qweight*(2**4)
                    #Qweight_l=Qweight
                    # #先将网络quantize到2nbite位
                    # delta = (cur_max - cur_min) / (self.bit_range_input_2n)
                    # interval = (Qactivation - cur_min) // delta
                    # mi = (interval + 0.5) * delta + cur_min
                    # # phi_function就是函数近似操作
                    # Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                    # # sgn就是round操作
                    # Qactivation = self.sgn(Qactivation)
                    # #取出用两个intn表示int2n的高位和低位
                    #这两步（//和%也就是下面三行代码）应该是离散操作不能求导，应该写到自定义的求导里面,
                    #这里写一个函数替代下面三行并自定义一个求导
                    #Qactivation_nbit_h, Qactivation_nbit_l=Qactivation_split.apply(Qactivation)
                """
                def Qactivation_split(Qactivation):
                    #函数中的求导应该置为int8经过一个卷积核的导数一致，也就是，输出对输入的导数应该是一样的（对Qactivation的导数置为1/（2**5），低位置为1/2）
                    input:Qactivation 2n_bit位量化的
                    return   Qactivation_nbit_l,Qactivation_nbit_h
                    #用这两个n_bit位通过两个卷积层来等效一个n_bit通过1个卷积层的结果
                """

                    # Qactivation_nbit_h=Qactivation//(2**4)#没有考虑可导性的问题
                    # Qactivation_nbit_l=Qactivation-Qactivation_nbit_h*(2**4)#没有考虑可导性的问题
                    # #Qactivation_nbit_l=Qactivation%(2**4)#没有考虑可导性的问题

                    # #反量化回fp32
                    # Qactivation_nbit_h= self.dequantize(Qactivation_nbit_h, cur_min, delta, interval)
                    # Qactivation_nbit_l=self.dequantize(Qactivation_nbit_h, cur_min, delta, interval)
                    # #使用F.conv2d可以改变weight的值,这里训练的Qweight_h=Qweight_l*(2**4),高位卷积的权重训练，Qweight_l=Qweight低位卷积的权重
                    # output_nbit_h= F.conv2d(Qactivation_nbit_h, Qweight_h, Qbias, self.stride, self.padding, self.dilation, self.groups)
                    # output_nbit_l= F.conv2d(Qactivation_nbit_l, Qweight_l, Qbias, self.stride, self.padding, self.dilation, self.groups)
                    # #output_2nbit=output_nbit_h+output_nbit_l#和torch.add的差别在哪里：速度不一样
                    # output_2nbit=torch.add(output_nbit_h,output_nbit_l)


                #else:到下面一个else之前都要缩进
                # bit_range_input_n是激活值低比特的范围，self.bit_range_input_2n是激活值原比特数两倍的范围
                # delta =  (cur_max - cur_min)/(self.bit_range_2n) 拆分之后，合并扩大到原来两倍的bit位数
                delta = (cur_max - cur_min)/(bit_range)#需要改动这里的bit数
                interval = (Qactivation - cur_min) //delta#这里不能求导，可以，因为里面不含参数
                mi = (interval + 0.5) * delta + cur_min
                #phi_function就是函数近似操作
                Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                #sgn就是round操作
                Qactivation = self.sgn(Qactivation)
                #反量化
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)
                "*********************** 近似round函数，然后round(量化)*************************"
            #DSQconv层中的只需要 给Qactivation, Qweight, Qbias
            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)#量化卷积

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output