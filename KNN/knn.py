from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
def file2Matrix(filename):#输入标准化(转变成算法能处理的输入格式)
    fr = open(filename)
    arraylines = fr.readlines()
    numberOfLines = len(arraylines)
    returnMat = zeros((numberOfLines, 3))  # 输入矩阵
    classLabelVector = []  # 标签矩阵
    index = 0
    for line in arraylines:
        lis = line.strip().split('\t')
        returnMat[index, :] = lis[0:3]
        classLabelVector.append(int(lis[-1]))
        index += 1
    return returnMat, classLabelVector  # 返回两个矩阵
def data_Show(datingMat,LabelsMat):#数据可视化分析
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingMat[:,0],datingMat[:,1],c=LabelsMat)
    plt.show()
def auto_Norm(datingMat):#数据归一化
    min = datingMat.min(0)
    max = datingMat.max(0)
    range = max - min
    normMat = zeros(shape(datingMat))
    m = datingMat.shape[0]
    normMat = datingMat - tile(min,(m,1))
    normMat = normMat/tile(range,(m,1))
    return normMat, range,min
def classify(inX,datingMat,Labels,k):#构造分类器
    data_size = datingMat.shape[0]#数据集行数
    diffMat = tile(inX,(data_size,1)) - datingMat
    sqdiffMat = diffMat**2
    distance = (sqdiffMat.sum(axis = 1))**0.5#计算距离,先算平方和再开根号 axis = 0是按列（跨行）求和 axis =1是按行（跨列）求和
    sorted_distance = distance.argsort()#返回的是数组值从小到大的索引值
    classcount = {}#字典
    for i in range(k):
        votelabel = Labels[sorted_distance[i]]#sorted_distance[i]是获取第i近的索引值，然后根据索引去Labels中找对应的标签
        classcount[votelabel] = classcount.get(votelabel,0)+1#把第K近的标签都添加到字典中并累加出现的标签次数
    print(classcount)
    sorted_classcount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)#python3中items()已取代iteritems()   key指定按哪个轴或者项进行排序
    return sorted_classcount[0][0]#注意下这两个函数 classcount.items()把字典变成可迭代的列表（字典项变成列表项） operator.itemgetter(1)获取第1个域值（从0开始）
def dating_Test(normMat,LabelsMat):#测试分类器
    hoRatio = 0.1#测试率（这里是10%）
    m = normMat.shape[0]
    test_vecs = int(m*hoRatio)#选择测试集
    errorcount = 0
    for i in range(test_vecs):
        classify_result = classify(normMat[i,:],normMat[test_vecs:m,:],LabelsMat[test_vecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classify_result,LabelsMat[i]))
        if (classify_result != LabelsMat[i]):
            errorcount += 1.0
    print("the total error rate is: %f" % (errorcount/float(test_vecs)))
def classifyPerson(datingMat,LabelsMat):#实际使用，即输入数据集中不存在的实例进行测试
    resultList = ['not at all','in small doses','in large doses']
    personTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    normMat, range, min = auto_Norm(datingMat)
    inArr = array([ffMiles,personTats,iceCream])
    classifyResult = classify((inArr-min)/range,normMat,LabelsMat,3)
    print("you will probably like this person: ",resultList[classifyResult-1])
def main():
    path = "datingTestSet2.txt"#数据集路径(默认在当前工作目录下)
    datingMat, LabelsMat = file2Matrix(path)
    #data_show(datingMat,LabelsMat)
    #normMat,range,min = auto_Norm(datingMat)
    #dating_Test(normMat,LabelsMat)
    classifyPerson(datingMat,LabelsMat)
main()