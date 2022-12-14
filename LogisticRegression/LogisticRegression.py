import matplotlib.pyplot as plt
import torch

def LoadData(Path_Name):
    #返回（ParamNum，X，Y）
    File = open(Path_Name)
    # Param_Num = Features_Num + 1
    Param_Num = len(File.readline().replace("\n", "").split(","))
    X, Y = list(list()), list(list())
    for Line in File.readlines():
        Data = list()
        Data.extend(list(map(float, Line.replace("\n", "").split(","))))
        Data.append(1)
        Y.append(list(Data[0:1]))
        X.append(list(Data[1:]))
    File.close()
    return [Param_Num, torch.tensor(X), torch.tensor(Y)]

def Sigmoid(Inputs):
    temp = torch.exp(Inputs)
    Result = temp / (1 + temp)
    # 除去nan值（因为inf/inf=nan，用1替代）
    for index,value in enumerate(Result):
        if value.item() != value.item(): Result[index][0]=1
    return Result

# def Log_Likelihood_Function(Inputs, Labels):
#     Part1 = Labels * torch.log(Inputs)
#     Part2 = (1 - Labels) * torch.log(1 - Inputs)
#     for index,value in enumerate(Labels):
#         if (value == 0): Part1[index] = 0
#         else: Part2[index] = 0
#     temp = Part1 + Part2
#     return temp.sum().item()

def Train(TrainSet, Epochs=100, Learning_Rate=0.00001, Batch_Size=36):
    print("Train:")
    ParamNum, Features, Labels = LoadData(TrainSet)
    AccRecoder = []
    # Init parameter
    Param = torch.randn((ParamNum, 1))*10
    # Begin Train
    DataNumber = Features.shape[0]
    for epoch in range(Epochs):
        if (epoch + 1) % 100 == 0:
            Learning_Rate = Learning_Rate / 2
        ACC, Start = 0, 0
        while Start < DataNumber:
            Stop = Start + Batch_Size
            Stop = DataNumber if Stop >= DataNumber else Stop
            Pred = Sigmoid(Features[Start:Stop] @ Param)
            for i in range(Start,Stop):
                label = 1 if Pred[i - Start] > 0.5 else 0
                if Labels[i]==label: ACC = ACC + 1
            # 使用极大似然估计和梯度下降的方法进行求解，使得对数似然函数达到最大值
            # Update Parameter
            Diffr = (Labels[Start:Stop] - Pred).sum()
            for index in range(Start, Stop):
                for num in range(Param.shape[0]):
                    Param[num][0] = Param[num][0] + Learning_Rate * Features[index][num] * Diffr
            Start = Start + Batch_Size
        AccRecoder.append(ACC / DataNumber * 100)
        print(f"\tepoch {epoch + 1}  Loss:{ACC / DataNumber * 100:0.2f}%")
    # Return Parameter,AccRecoder
    return Param, AccRecoder

if __name__ == "__main__":
    epoch = 50
    lr = 0.00005
    bs = 64
    NewParam, Acc = Train("Train.csv", epoch, lr, bs)
    X = [i + 1 for i in range(epoch)]
    plt.figure()
    plt.ylabel("Acc")
    plt.xlabel("Epoch")
    plt.plot(X, Acc)
    plt.savefig("Acc_Photo")
