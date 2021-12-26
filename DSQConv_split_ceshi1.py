import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundWithGradient(torch.autograd.Function):
    #这里应该就是对缩放的ψ函数进行还原到[-1,1]（这里其实是因为ψ函数才是对应的近似形态）——>再round——>最后让数只能在[0,1]两个值之中选
    @staticmethod
    def forward(ctx, x):#这样操作,会使梯度下降那里出现问题，因为forward里面的操作没有记录梯度，但是影响因应该不大(平移不影响梯度，缩放也不大，因为alpha只是微调范围，而后面乘以2也不大)
        delta = torch.max(x) - torch.min(x)#写在自定义函数中是因为写在里面才能梯度下降alpha，如果写下外面，先乘再除是一个无用操作
        #ctx.save_for_backward(delta)
        x = (x / delta + 0.5)#这里加0.5是想让round之后保持在[0,1]这两个取值中的一个其实和后面dequantize函数结合就是让正整数N的相邻区间[N,N+1]内的小数归并到[N,N+1]
        return x.round() * 2 - 1


    @staticmethod
    def backward(ctx, g):
        #delta,=ctx.saved_variables
        #return (torch.div(g,delta))*2
        return g

"************************自定义通道合并的//和%不可导的导数**********************************"
class Qactivation_split(torch.autograd.Function):#定义等效梯度
    @staticmethod
    #forward(ctx,Qactivation,num_bit_input) 后期通道合并更低比特时如两个2bit合4bit，（2**4）变为（2**2），这个num_bit_input就是合成前bit位数
    def forward(ctx,Qactivation):#后期forward这还要输入一个低bit的range
        Qactivation_nbit_h = Qactivation // (2 ** 4)
        Qactivation_nbit_l = Qactivation % (2 ** 4)

        return Qactivation_nbit_h,Qactivation_nbit_l
    @staticmethod
    def backward(ctx, g_nbit_h,g_nbit_l):
        #return g_nbit_h*(1/((2**4)))*0.8+g_nbit_l**0.2
        #return (g_nbit_h+g_nbit_l)/(2**4+1)
        return g_nbit_h*(1/((2**5)))+g_nbit_l*(1/2)

class Qactivation_split_with_grad_1(torch.autograd.Function):#不定义等效梯度
    @staticmethod
    #forward(ctx,Qactivation,num_bit_input) 后期通道合并更低比特时如两个2bit合4bit，（2**4）变为（2**2），这个num_bit_input就是合成前bit位数
    def forward(ctx,Qactivation):#后期forward这还要输入一个低bit的range
        Qactivation_nbit_h = Qactivation // (2 ** 4)
        Qactivation_nbit_l = Qactivation % (2 ** 4)

        return Qactivation_nbit_h,Qactivation_nbit_l
    @staticmethod
    def backward(ctx, g_nbit_h,g_nbit_l):
        return g_nbit_h+g_nbit_l



# class Qactivation_split_interval((torch.autograd.Function)):
#     def forward(ctx, interval):
#         interval_nbit_h = interval // (2 ** 4)
#         interval_nbit_l = interval% (2 ** 4)
#
#         return interval_nbit_h, interval_nbit_l
#
#     @staticmethod
#     def backward(ctx, g_nbit_h, g_nbit_l):
#         # return g_nbit_h*(1/((2**4)))*0.8+g_nbit_l**0.2
#         # return (g_nbit_h+g_nbit_l)/(2**4+1)
#         return g_nbit_h * (1 / ((2 ** 5))) + g_nbit_l * (1 / 2)

# class Qactivation_split_h(torch.autograd.Function):#
#     @staticmethod
#     #forward(ctx,Qactivation,num_bit_input) 后期通道合并更低比特时如两个2bit合4bit，（2**4）变为（2**2），这个num_bit_input就是合成前bit位数
#     def forward(ctx,Qactivation):#后期forward这还要输入一个低bit的range
#         Qactivation_nbit_h = Qactivation // (2 ** 4)
#
#
#         return Qactivation_nbit_h
#     @staticmethod
#     def backward(ctx, g_nbit_h):
#
#         return (g_nbit_h)/(2**5)
#
#
# class Qactivation_split_l(torch.autograd.Function):  #
#     @staticmethod
#     # forward(ctx,Qactivation,num_bit_input) 后期通道合并更低比特时如两个2bit合4bit，（2**4）变为（2**2），这个num_bit_input就是合成前bit位数
#     def forward(ctx, Qactivation):  # 后期forward这还要输入一个低bit的range
#         Qactivation_nbit_l = Qactivation % (2 ** 4)
#
#         return Qactivation_nbit_l
#
#     @staticmethod
#     def backward(ctx, g_nbit_l):
#         # return g_nbit_h*(1/((2**5)*(2**4)))+g_nbit_l*(1/2)
#         return (g_nbit_l) / 2


class conv_add(torch.autograd.Function):
    @staticmethod#后期forward这还要输入一个低bit的range
    def forward(ctx,output_nbit_h,output_nbit_l):
        output_2nbit = torch.add(output_nbit_h, output_nbit_l)
        return output_2nbit

    @staticmethod
    def backward(ctx, g):
        return g,g
        #return g*(2**4),g


'******************这里没有用文章中的训练边界的方法，直接用最简单粗暴的量化反量化来解决问题 ******'
class quantize_dequantize(torch.autograd.Function):
    @staticmethod  # 后期forward这还要输入一个低bit的range
    def forward(ctx, x,num_bit):
        max_x=torch.max(x)
        min_x=torch.min(x)
        delta=(max_x-min_x)/(2**num_bit-1)

        return (x/delta).round()* delta

    @staticmethod
    def backward(ctx, g):
        return g, None

'******************这里没有用文章中的量化方法，直接用EWGS量化方法进行训练******'
class quantize_dequantizewithEWGS(torch.autograd.Function):
    @staticmethod  # 后期forward这还要输入一个低bit的range
    def forward(ctx, x,num_bit):
        max_x=torch.max(x)
        min_x=torch.min(x)
        delta=(max_x-min_x)/(2**num_bit-1)
        x_out=(x / delta).round() * delta
        ctx.save_for_backward(x - x_out)
        return x_out

    @staticmethod
    def backward(ctx, g):
        diff,=ctx.saved_tensors
        scale = 1 + 0.001 * torch.sign(g) * diff
        return g*scale, None


#后面DSQConv_split增加一个flag，来进行对拆分后的高低位进行量化反量化的方法选择，现在初步有原始的量化反量化，EWGS两种
class DSQConv_split(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 momentum=0.1,
                 num_bit_w=8,num_bit_input=8,flag_split=False,flag_quant_method="Origin", QInput=True, bSetQ=True):  # bSetQ = True
        super(DSQConv_split, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit_w = num_bit_w
        self.num_bit_input=num_bit_input
        self.quan_input = QInput
        self.flag_split=flag_split#测试一下为flase的情况
        #self.bit_range_w = 2 ** self.num_bit_w - 1  # 计算比特数
        #self.bit_range_input=2 ** self.num_bit_input - 1
        self.is_quan = bSetQ
        self.flag_quant_method=flag_quant_method
        self.momentum = momentum
        if self.is_quan:
            #print(self.quan_input)
            # using int32 max/min as init and backprogation to optimization#32位整数
            # Weight
            self.uW = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
            self.lW = nn.Parameter(data=torch.tensor((-1) * (2 ** 32)).float())
            self.register_buffer('running_uw', torch.tensor([self.uW.data]))  # init with uw
            self.register_buffer('running_lw', torch.tensor([self.lW.data]))  # init with lw
            self.alphaW = nn.Parameter(data=torch.tensor(0.2).float())
            # Bias
            if self.bias is not None:
                self.uB = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
                self.lB = nn.Parameter(data=torch.tensor((-1) * (2 ** 32)).float())
                self.register_buffer('running_uB', torch.tensor([self.uB.data]))  # init with ub
                self.register_buffer('running_lB', torch.tensor([self.lB.data]))  # init with lb
                self.alphaB = nn.Parameter(data=torch.tensor(0.2).float())

            # Activation input
            if self.quan_input:
                # 写在这里train中的set函数不起作用，因为在set之前 transformer.register进行层替换时已经进行了初始化，应该在foward中根据set_quantinput中设置的True和False来进行初始化（set函数才起作用
                self.uA = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
                self.lA = nn.Parameter(data=torch.tensor((-1) * (2 ** 32)).float())
                self.register_buffer('running_uA', torch.tensor([self.uA.data]))  # init with uA
                self.register_buffer('running_lA', torch.tensor([self.lA.data]))  # init with lA
                self.alphaA = nn.Parameter(data=torch.tensor(0.2).float())
            # self.uA_h = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
            # self.lA_h = nn.Parameter(data=torch.tensor((-1) * (2 ** 32)).float())
            # self.register_buffer('running_uA_h', torch.tensor([self.uA_h.data]))  # init with uA
            # self.register_buffer('running_lA_h', torch.tensor([self.lA_h.data]))  # init with lA
            self.alphaA_h = nn.Parameter(data=torch.tensor(0.2).float())
            # self.uA_l = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
            # self.lA_l = nn.Parameter(data=torch.tensor((-1) * (2 ** 32)).float())
            # self.register_buffer('running_uA_l', torch.tensor([self.uA_l.data]))  # init with uA
            # self.register_buffer('running_lA_l', torch.tensor([self.lA_l.data]))  # init with lA
            self.alphaA_l = nn.Parameter(data=torch.tensor(0.2).float())

    def clipping(self, x, upper, lower):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x

    def phi_function(self, x, mi, alpha, delta):

        # alpha should less than 2 or log will be None
        # alpha = alpha.clamp(None, 2)
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
        s = 1 / (1 - alpha)
        k = (2 / alpha - 1).log() * (1 / delta)
        x = (((x - mi) * k).tanh()) * s
        return x

    def phi_function_with_alpha(self, x, mi, alpha, delta):

        # alpha should less than 2 or log will be None
        # alpha = alpha.clamp(None, 2)
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
        s = 1 / (1 - alpha)
        k = (2 / alpha - 1).log() * (1 / delta)
        x1=(((x - mi) * k).tanh())
        x= x1 * s
        return x,k,s,x1

    def phi_function_no_alpha(self, x, mi, delta):

        # alpha should less than 2 or log will be None
        # alpha = alpha.clamp(None, 2)
        #alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
        k = 100
        s=1/((k*0.5*delta).tanh())
        x = (((x - mi) * k).tanh()) * s
        return x

    def sgn(self, x):#这个函数不知到有什么用
        x = RoundWithGradient.apply(x)

        return x

    def dequantize(self, x, lower_bound, delta, interval):

        # save mem
        x = ((x + 1) / 2 + interval) * delta + lower_bound

        return x
    def dequantize_h(self, x_1, lower_bound, delta, interval):#x_1=(x + 1) / 2

        # save mem
        x = (x_1+ interval) * delta + lower_bound/(2**4)

        return x
    def dequantize_l(self, x_1, lower_bound, delta, interval):#x_1=x_1=(x + 1) / 2

        # save mem
        x = ( x_1+ interval) * delta + lower_bound

        return x






    def forward(self, x):#重写了conv2d的forward函数，conv2d的forward函数就是调用的f.conv2d
        bit_range_w=2 ** self.num_bit_w - 1
        bit_range_input = 2 ** self.num_bit_input - 1
        bit_range_input_add=2 ** (self.num_bit_input/2) - 1
        #print(bit_range_input)
        if self.is_quan:
            # Weight Part
            # moving average
            if self.training:
                cur_running_lw = self.running_lw.mul(1 - self.momentum).add((self.momentum) * self.lW)
                cur_running_uw = self.running_uw.mul(1 - self.momentum).add((self.momentum) * self.uW)
            else:
                cur_running_lw = self.running_lw
                cur_running_uw = self.running_uw

            Qweight = self.clipping(self.weight, cur_running_uw, cur_running_lw)
            cur_max = torch.max(Qweight)
            cur_min = torch.min(Qweight)
            delta = (cur_max - cur_min) / (bit_range_w)
            interval = (Qweight - cur_min) // delta
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.alphaW, delta)
            #Qweight=Qweight/(torch.max(Qweight)-torch.min(Qweight))+0.5
            Qweight = self.sgn(Qweight)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval)

            Qbias = self.bias
            # Bias
            if self.bias is not None:
                # self.running_lB.mul_(1-self.momentum).add_((self.momentum) * self.lB)
                # self.running_uB.mul_(1-self.momentum).add_((self.momentum) * self.uB)
                if self.training:
                    cur_running_lB = self.running_lB.mul(1 - self.momentum).add((self.momentum) * self.lB)
                    cur_running_uB = self.running_uB.mul(1 - self.momentum).add((self.momentum) * self.uB)
                else:
                    cur_running_lB = self.running_lB
                    cur_running_uB = self.running_uB

                Qbias = self.clipping(self.bias, cur_running_uB, cur_running_lB)
                cur_max = torch.max(Qbias)
                cur_min = torch.min(Qbias)
                delta = (cur_max - cur_min) / (bit_range_w)#weight的bit
                interval = (Qbias - cur_min) // delta
                mi = (interval + 0.5) * delta + cur_min
                Qbias = self.phi_function(Qbias, mi, self.alphaB, delta)
               # Qbias = Qweight / (torch.max(Qbias) - torch.min(Qbias)) + 0.5
                Qbias = self.sgn(Qbias)
                Qbias = self.dequantize(Qbias, cur_min, delta, interval)

            # Input(Activation)
            Qactivation = x
            if self.quan_input:

                if self.training:
                    cur_running_lA = self.running_lA.mul(1 - self.momentum).add((self.momentum) * self.lA)
                    cur_running_uA = self.running_uA.mul(1 - self.momentum).add((self.momentum) * self.uA)
                else:
                    cur_running_lA = self.running_lA
                    cur_running_uA = self.running_uA

                Qactivation = self.clipping(x, cur_running_uA, cur_running_lA)
                cur_max = torch.max(Qactivation)
                cur_min = torch.min(Qactivation)
                if self.flag_split:
                    #print( self.flag_split)
                    Qweight_h=Qweight*(2**4)
                    Qweight_l=Qweight
                    #先将网络quantize到2nbit位
                    delta = (cur_max - cur_min) / (bit_range_input)#这里是8bit的量化


                    "********应该是对interval这里进行量化应该才是对的*************"

                    interval = (Qactivation - cur_min) // delta#量化的体现#这里截断了
                    #interval的存储形式是interval_nbit_h, interval_nbit_l
                    #interval_nbit_h, interval_nbit_l = Qactivation_split_interval.apply(interval)
                    interval_nbit_h, interval_nbit_l=interval//(2**4),interval%(2**4)#这里求余数，虽然有梯度但是被前面求interval的整除截断了
                    mi = (interval + 0.5) * delta + cur_min
                    # phi_function就是函数近似操作

                    Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                    #Qactivation=Qactivation/ (torch.max(Qactivation) - torch.min(Qactivation)) + 0.5

                    "********训练时可能是sgn这有问题，因为这里好像不是表示映射到int8 interval才是映射到int8的值，代码已解决***********"

                    Qactivation = self.sgn(Qactivation)#就是取整操作
                    Qactivation=(Qactivation+1)/2
                    #Qactivation_nbit_h=Qactivation_split_h.apply(Qactivation)
                    #Qactivation_nbit_l=Qactivation_split_l.apply(Qactivation)
                    Qactivation_nbit_h, Qactivation_nbit_l=Qactivation_split.apply(Qactivation)
                    #反量化回fp32
                    Qactivation_nbit_h= self.dequantize_h(Qactivation_nbit_h, cur_min, delta, interval_nbit_h)
                    Qactivation_nbit_l=self.dequantize_l(Qactivation_nbit_l, cur_min, delta, interval_nbit_l)
                    #使用F.conv2d可以改变weight的值,这里训练的Qweight_h=Qweight_l*(2**4),高位卷积的权重训练，Qweight_l=Qweight低位卷积的权重
                    output_nbit_h= F.conv2d(Qactivation_nbit_h, Qweight_h, Qbias, self.stride, self.padding, self.dilation, self.groups)
                    output_nbit_l= F.conv2d(Qactivation_nbit_l, Qweight_l, Qbias, self.stride, self.padding, self.dilation, self.groups)

                    "*************************不对output_nbit_h,output_nbit_l进行量化*****************************************"
                    # if self.flag_quant_method is None:
                    #     output_2nbit = output_nbit_h + output_nbit_l
                    #     return output_2nbit

                    "*************************原始量化反量化对output_nbit_h,output_nbit_l进行量化*****************************************"
                    # print(self.flag_quant_method)
                    # print(self.flag_quant_method == "Origin")
                    if self.flag_quant_method is "Origin":
                         output_nbit_h = quantize_dequantize.apply(output_nbit_h,bit_range_input/2)
                         output_nbit_l=quantize_dequantize.apply(output_nbit_l,bit_range_input/2)
                         output_2nbit = output_nbit_h + output_nbit_l
                         return output_2nbit

                    "*************************EWGS对对output_nbit_h,output_nbit_l进行量化*************************************"
                    if self.flag_quant_method == "EWGS":
                        #print(1)
                        output_nbit_h = quantize_dequantizewithEWGS.apply(output_nbit_h, bit_range_input / 2)
                        output_nbit_l = quantize_dequantizewithEWGS.apply(output_nbit_l, bit_range_input / 2)
                        output_2nbit = output_nbit_h + output_nbit_l
                        return output_2nbit

                    output_2nbit = output_nbit_h + output_nbit_l  # 这里就是直接高比特和低比特相加没有量化和反量化
                    # output_2nbit=output_nbit_h+output_nbit_l#和torch.add的差别在哪里：速度不一样
                    # output_2nbit=conv_add.apply(output_nbit_h,output_nbit_l)
                    "*******************DSQ量化output_nbit_h,output_nbit_l需要量化然后反量化到4bit上去，来模拟量化误差***************"
                    # if self.flag_quant_method is "DSQ":
                    # cur_running_lA_h = self.running_lA_h.mul(1 - self.momentum).add((self.momentum) * self.lA_h)
                    # cur_running_uA_h = self.running_uA_h.mul(1 - self.momentum).add((self.momentum) * self.uA_h)
                    # cur_running_lA_l = self.running_lA_l.mul(1 - self.momentum).add((self.momentum) * self.lA_l)
                    # cur_running_uA_l= self.running_uA_l.mul(1 - self.momentum).add((self.momentum) * self.uA_l)
                    # # Qactivation_h = self.clipping(output_nbit_h, cur_running_uA_h, cur_running_lA_h)
                    # # Qactivation_l = self.clipping(output_nbit_l, cur_running_uA_l, cur_running_lA_l)
                    # #
                    # # # "***************************** 对高位和低位进行4bit的量化和反量化 ******************************"
                    # # #
                    # # #
                    # # #
                    # # #
                    # # # "**********************************设计反向传播让其前向是quantize-dequantize到int4 这个过程让其反向传播的梯度为1************************************* "
                    # # "在上面两个卷积输出的值加起来其实就满足了所有的梯度等效，但是加了量化反量化之后梯度就可能不满足了，需要手动定义forward的输出对两个卷积的输出的梯度都为1才能满足等效梯度"
                    # # "但是DSQ量化反量化会多6个参数alpha（两个int4分别1个），以及量化前输出的量化边界（两个int4分别2个）"
                    # #
                    # # "尝试过程中也可以将我的那个roundwithgrident改回来作者那样写"
                    # # "尝试将alpha加入到forward参数中 "
                    # # "有一个没有NAN但是结果精度很低的那一个，情况因该就是作者那个roundgridengt加上3参数或者时一个参数（一个参数的应该就是把clip放里面的情况的情况，alpha那里写的是self.alpha"
                    # #"精度低是这个def forward(ctx,output_nbit_h,cur_running_uA_h, cur_running_lA_h):然后rongdgrident是用的源代码的"
                    # #
                    # # 直接用最简单的量化反量化
                    # # 量化反量化的基础上加上EWGS或者LSQ或者两个都弄上

                    "*************************************************这里对phi函数使用alpha，导数实在是太复杂了******************************************** "
                    # class quntize_dequntize_with_alpha(torch.autograd.Function):
                    #     @staticmethod  # 后期forward这还要输入一个低bit的range
                    #     #def forward(ctx,output_nbit_h,cur_running_uA_h, cur_running_lA_h,lA_h,uA_h):#将这个
                    #     #def forward(ctx,output_nbit_h,cur_running_uA_h, cur_running_lA_h):#这种方式是可以跑的
                    #     def forward(ctx, output_nbit_h,alphaA_h):#这里直接求最大最小没有训练边界
                    #     #def forward(ctx, Qactivation_h):
                    #         # cur_running_lA_h = cur_running_lA_h.mul(1 - 0.1).add((0.1) * lA_h)
                    #         # cur_running_uA_h = cur_running_uA_h.mul(1 - 0.1).add((0.1) * uA_h)
                    #
                    #         Qactivation_h_grad = output_nbit_h  # 直接求最大最小没有求边界
                    #         #Qactivation_h = self.clipping(output_nbit_h, cur_running_uA_h, cur_running_lA_h)
                    #
                    #         cur_max_h = torch.max(Qactivation_h_grad )
                    #         cur_min_h = torch.min(Qactivation_h_grad )
                    #         delta_h = (cur_max_h - cur_min_h) / (bit_range_input_add)
                    #         interval_h = (output_nbit_h - cur_min_h) // delta_h
                    #         mi_h = (interval_h + 0.5) * delta_h + cur_min_h
                    #         Qactivation_h = self.phi_function(Qactivation_h_grad , mi_h, self.alphaA_h, delta_h)
                    #         Qactivation_h = self.sgn(Qactivation_h)#这里其实就是量化的体现，因为phi_function函数将数据都趋近于-1和1
                    #
                    #
                    #
                    #         "对interval+Qactivation_h这里可以看成量化后的值，然后再通过dequantize反量化对这里可以设置的梯度1/s,这个想法应该是正确的避开alpha"
                    #         "学习如何成为一个    "
                    #         Qactivation_h = self.dequantize(Qactivation_h, cur_min_h, delta_h, interval_h)
                    #         ctx.save_for_backward(Qactivation_h_grad,mi_h,self.alphaA_h,delta_h )
                    #         return Qactivation_h
                    #     @staticmethod
                    #     def backward(ctx, g_o):
                    #         #return g_o,None,None,g_o,g_o
                    #         #return g_o,g_o,g_o
                    #         return g_o,g_o
                    #         #return g_o
                    #         # return g*(2**4),g

                    " ******************************************这里弄phi函数没有alpha的********************************************************                     "
                    # class quntize_dequntize_without_alpha(torch.autograd.Function):
                    #     @staticmethod  # 后期forward这还要输入一个低bit的range
                    #     #def forward(ctx,output_nbit_h,cur_running_uA_h, cur_running_lA_h,lA_h,uA_h):#将这个
                    #     #def forward(ctx,output_nbit_h,cur_running_uA_h, cur_running_lA_h):
                    #     def forward(ctx, output_nbit_h):#这里直接求最大最小没有训练边界
                    #     #def forward(ctx, Qactivation_h):
                    #         # cur_running_lA_h = cur_running_lA_h.mul(1 - 0.1).add((0.1) * lA_h)
                    #         # cur_running_uA_h = cur_running_uA_h.mul(1 - 0.1).add((0.1) * uA_h)
                    #
                    #         Qactivation_h = output_nbit_h  # 直接求最大最小没有求边界
                    #         #Qactivation_h = self.clipping(output_nbit_h, cur_running_uA_h, cur_running_lA_h)
                    #
                    #         cur_max_h = torch.max(Qactivation_h)
                    #         cur_min_h = torch.min(Qactivation_h)
                    #         delta_h = (cur_max_h - cur_min_h) / (bit_range_input_add)
                    #         interval_h = (output_nbit_h - cur_min_h) // delta_h
                    #         mi_h = (interval_h + 0.5) * delta_h + cur_min_h
                    #         Qactivation_h = self.phi_function_no_alpha(Qactivation_h, mi_h, delta_h)
                    #         Qactivation_h = self.sgn(Qactivation_h)#这里其实就是量化的体现，因为phi_function函数将数据都趋近于-1和1
                    #
                    #
                    #
                    #         "对interval+Qactivation_h这里可以看成量化后的值，然后再通过dequantize反量化对这里可以设置的梯度1/s,这个想法应该是正确的避开alpha"
                    #         "学习如何成为一个    "
                    #         Qactivation_h = self.dequantize(Qactivation_h, cur_min_h, delta_h, interval_h)
                    #
                    #         return Qactivation_h
                    #     @staticmethod
                    #     def backward(ctx, g_o):
                    #         #return g_o,None,None,g_o,g_o
                    #         #return g_o,g_o,g_o
                    #
                    #         return g_o
                    #         # return g*(2**4),g

                    # # #
                    # # # " 不学习边界"
                    # # Qactivation_h = quntize_dequntize.apply(output_nbit_h，self.alphaA_h)
                    # # Qactivation_h = quntize_dequntize.apply(output_nbit_l，self.alphaA_l)
                    # # Qactivation_l=quntize_dequntize.apply(output_nbit_l)

                    # Qactivation_h=quntize_dequntize_without_alpha.apply(output_nbit_h)
                    # Qactivation_l=quntize_dequntize_without_alpha.apply(output_nbit_l)
                    # # Qactivation_h = quntize_dequntize.apply(Qactivation_h)
                    # # Qactivation_l=quntize_dequntize.apply(Qactivation_l)
                    # # "学习边界"
                    # # # Qactivation_h=quntize_dequntize.apply(output_nbit_h,self.running_uA_h,self.running_lA_h,self.lA_h,self.uA_h)
                    # # # Qactivation_l=quntize_dequntize.apply(output_nbit_l, self.running_uA_l, self.running_lA_l,self.lA_l,self.uA_l)
                    # Qactivation_h = quntize_dequntize_h.apply(output_nbit_h,cur_running_uA_h,cur_running_lA_h)
                    # Qactivation_l = quntize_dequntize_l.apply(output_nbit_l, cur_running_uA_l, cur_running_lA_l)
                    #
                    # # #
                    "**********************************设计反向传播让其前向是quantize-dequantize到int4 这个过程让其反向传播的梯度为正常流动的梯度******************** "
                    # Qactivation_h = output_nbit_h  # 直接求最大最小没有求边界
                    # #Qactivation_h = self.clipping(output_nbit_h, cur_running_uA_h, cur_running_lA_h)
                    # cur_max_h = torch.max(Qactivation_h)
                    # cur_min_h = torch.min(Qactivation_h)
                    # delta_h = (cur_max_h - cur_min_h) / (bit_range_input_add)
                    # interval_h = (output_nbit_h - cur_min_h) // delta_h
                    # mi_h = (interval_h + 0.5) * delta_h + cur_min_h
                    # Qactivation_h = self.phi_function_no_alpha(Qactivation_h, mi_h, delta_h)
                    # Qactivation_h = self.sgn(Qactivation_h)
                    # Qactivation_h = self.dequantize(Qactivation_h, cur_min_h, delta_h, interval_h)

                    # #Qactivation_l = self.clipping(output_nbit_l, cur_running_uA_l, cur_running_lA_l)
                    # Qactivation_l = output_nbit_l
                    # cur_max_l = torch.max(Qactivation_l)
                    # cur_min_l = torch.min(Qactivation_l)
                    # delta_l = (cur_max_l - cur_min_l) / (bit_range_input_add)
                    # interval_l = (output_nbit_l - cur_min_l) // delta_l
                    # mi_l = (interval_l + 0.5) * delta_l + cur_min_l
                    # Qactivation_l  = self.phi_function_no_alpha(Qactivation_l , mi_l, delta_l)

                    # Qactivation_l = self.sgn(Qactivation_l)
                    # Qactivation_l = self.dequantize(Qactivation_l, cur_min_l, delta_l, interval_l)
                    # output_2nbit=torch.add(Qactivation_h,Qactivation_l)
                    "***************************** 对高位和低位进行4bit的量化和反量化 ******************************"
                    return output_2nbit



                delta = (cur_max - cur_min) / (bit_range_input)  # input（activation）的比特数
                interval = (Qactivation - cur_min) // delta
                mi = (interval + 0.5) * delta + cur_min
                Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                #Qactivation = Qactivation / (torch.max(Qactivation) - torch.min(Qactivation)) + 0.5
                Qactivation = self.sgn(Qactivation)
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)
            #DSQconv层中的只需要 给Qactivation, Qweight, Qbias
            output = F.conv2d(Qactivation, Qweight, Qbias, self.stride, self.padding, self.dilation, self.groups)

        else:
            output = F.conv2d(x, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        return output