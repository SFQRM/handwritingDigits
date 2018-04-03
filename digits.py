"""
    示例：手写识别系统
"""
from numpy import *
import operator
from os import listdir

"""
    实施kNN分类算法
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]      # shape方法是numpy的函数，shape[0]是第二维的长度
    # 距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet      # tile(A,n)函数是numpy的函数，功能是将数组A重复n次
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)     # 没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加
    distances = sqDistances**0.5
    # 距离计算
    sortedDistIndicies = distances.argsort()        # argsort()函数是numpy的函数，函数返回的数组值从小到大的索引值
    classCount = {}     # 初始化空字典
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # dict[k] = v 将值v关联到键k上
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1       # 字典get()函数返回指定键的值，如果值不在字典中返回默认值。
    # 选择距离最小的k个点
    # classCount.iteritems()是将classCount字典分解为元组列表
    # sorted()方法返回的是一个新的列表
    sortedClassCount = sorted(classCount.items(),           # 要排序的对象
                              key=operator.itemgetter(1),   # 指定取待排序元素的第二项进行排序
                              reverse=True)                 # 降序
    return sortedClassCount[0][0]

"""
    Step1——收集数据：略
    Step2——准备数据：将图像转换为测试向量
"""
# 将图像格式转换为分类器使用的向量格式
def img2vector(filename):
    returnVect = zeros((1, 1024))       # 初始化一个1*1024的零矩阵
    fr = open(filename)     # 打开文件
    for i in range(32):
        lineStr = fr.readline()     # 每次读入一行
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])     # 将每行的头32个字符存储在数组中
    return returnVect

# testVector = img2vector('testDigits/0_13.txt')
# print(testVector[0, 0:31])      # 第0行的0~31个元素

"""
    Step3——分析数据：略
    Step4——训练算法：略
    Setp5——测试算法：使用k-近邻算法识别手写数字
"""
# 手写数字识别系统的测试
def handwritingClassTest():
    hwLabels = []       # 初始化标签向量
    trainingFileList = listdir('trainingDigits')        # 将trainingDigits文件夹中的文件列入
    m = len(trainingFileList)       # 得到trainDigits中文件个数
    trainingMat = zeros((m, 1024))      # 构建m*1024的零矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]       # 第i个文件的文件名
        fileStr = fileNameStr.split('.')[0]     # 以“.”将文件名分割为两部分，并取第一部分
        classNumStr = int(fileStr.split('_')[0])        # 从文件名解析分类数字
        hwLabels.append(classNumStr)        # 向标签向量中加入数字
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)      # 得到训练数据向量
    testFileList = listdir('testDigits')        # 将trainingDigits文件夹中的文件列入
    errorCount = 0.0
    mTest = len(testFileList)       # 得到trainingDigits中文件个数
    for i in range(mTest):
        fileNameStr = testFileList[i]       # 第i个文件的文件名
        fileStr = fileNameStr.split('.')[0]     # 以“.”将文件名分割为两部分，并取第一部分
        classNumStr = int(fileNameStr.split('_')[0])        # 从文件名解析分类数字
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)       # 得到输入向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with：%d, real answer is：%d"
              %(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is：%d" %errorCount)
    print("\nthe total error rate is：%f" %(errorCount/float(mTest)))

"""
    执行
"""
print(handwritingClassTest())
