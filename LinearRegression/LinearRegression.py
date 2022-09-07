import random
import torch

def Train(TrainSet, epochs, lr, batch_size):
    print("Train:")
    File = open(TrainSet)
    # Init parameter
    ParamNum = len(File.readline().replace("\n", "").split(","))
    Param = list()
    for _ in range(ParamNum): Param.append(random.uniform(1, 50))
    Param = torch.Tensor(Param)
    Param = Param.unsqueeze(dim=1)
    # Load Data
    X, Y = list(list()), list(list())
    for Line in File.readlines():
        Data = list()
        Data.extend(list(map(float, Line.replace("\n", "").split(","))))
        Data.append(1)
        Y.append(list(Data[0:1]))
        X.append(list(Data[1:]))
    X, Y = torch.Tensor(X), torch.Tensor(Y)
    File.close()
    # Begin Train
    DataNumber = X.shape[0]
    for epoch in range(epochs):
        AllLoss = 0
        Start = 0
        while Start < DataNumber:
            Stop = Start + batch_size
            Stop = DataNumber if Stop >= DataNumber else Stop
            Pred = X[Start:Stop] @ Param
            AllLoss = AllLoss + (0.5 * ((Pred - Y[Start:Stop]) ** 2)).sum().item()
            # Update Parameter
            Diffr = (Pred - Y[Start:Stop]).sum() / (Stop-Start)
            for index in range(Start, Stop):
                for num in range(Param.shape[0]):
                    Param[num][0] = Param[num][0] - lr * X[index][num] * Diffr
            Start = Start + batch_size
        print(f"epoch {epoch+1}  Loss:{AllLoss/DataNumber/10000}")
    # Return Parameter
    return Param

def Test(TestSet, Param):
    print("Test:")
    File = open(TestSet)
    File.readline()
    # Load Data
    X, Y = list(list()), list(list())
    for Line in File.readlines():
        Data = list()
        Data.extend(list(map(float, Line.replace("\n", "").split(","))))
        Data.append(1)
        Y.append(list(Data[0:1]))
        X.append(list(Data[1:]))
    X, Y = torch.Tensor(X), torch.Tensor(Y)
    File.close()
    # Begin Test
    DataNumber = X.shape[0]
    AllLoss = (0.5 * ((X @ Param - Y) ** 2)).sum().item()
    print(f"\t  Loss:{AllLoss / DataNumber / 10000}")

if __name__ == "__main__":
    Param = Train("Train.csv", 100, 0.000000000005, 10)
    Test("Test.csv",Param)