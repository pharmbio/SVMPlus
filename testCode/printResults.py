from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt



"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt
if 1:
    N = 4
    svmX = (0.015103, 0.272146, 0.285467, 0.260612)
    svmXStar = (0.000839, 0.092754, 0.127245  , 0.098628 )
    svmPlus = (0.013733, 0.226047  , 0.283943 , 0.243199  )
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, svmX , width, color='pink')

    rects2 = ax.bar(ind + width, svmXStar , width, color='y')

    rects3 = ax.bar(ind + width+width, svmPlus , width, color='g')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Observed Fuzziness')
    plt.ylim((0, .3))
    #ax.set_title('Comparision of efficiency')
    ax.set_xticks(ind + width )
    ax.set_xticklabels(('MNIST', 'AHR', 'HANSEN', 'MMP'))

    ax.legend((rects1[0], rects2[0], rects3[0]), ('SVM on X', 'SVM on X*', 'SVM on X and X* as PI'))


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)

    plt.show()




if 0:
    N = 4
    svmX = (.939125, 0.888889, .669124, .847522)
    svmXStar = (.987375, .917857, .809370, 0.896726)
    svmPlus = (.942875, .888889, .676651, 0.849292)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, svmX , width, color='pink')

    rects2 = ax.bar(ind + width, svmXStar , width, color='y')

    rects3 = ax.bar(ind + width+width, svmPlus , width, color='g')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Prediction Accuracy')
    plt.ylim((.55, 1))
    #ax.set_title('Comparision of predictive performance')
    ax.set_xticks(ind + width )
    ax.set_xticklabels(('MNIST', 'AHR', 'HANSEN', 'MMP'))

    ax.legend((rects1[0], rects2[0], rects3[0]), ('SVM on X', 'SVM on X*', 'SVM on X and X* as PI'))


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)

    plt.show()

if 0:
    # Comparing validiy and obs-fuzz using SVM and SVMPlus
    t = PrettyTable(['Dataset-AHR', 'M/L Method', 'Validity', 'Obs-fuzziness'])
    t.add_row(['MorganUnhashed', 'SVM', '', ''])
    t.add_row(['physchem', 'SVM', '.17', '.26'])
    t.add_row(['physchem+MorganUnhashed', 'SVM+', '.12', '22'])
    print(t)



'''
# SVM and SVM+ comparision with reduced dataset
t = PrettyTable(['Dataset-AHR', 'Method', 'Pred-accuracy'])
t.add_row(['MorganUnhashed', 'SVM', '0.850'])
t.add_row(['physchem', 'SVM', '0.680'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.706'])
print(t)

t = PrettyTable(['Dataset-MMP', 'Method', 'Pred-accuracy'])
t.add_row(['MorganUnhashed', 'SVM', '0.904'])
t.add_row(['physchem', 'SVM', '0.849'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.867'])
print(t)

t = PrettyTable(['Dataset-HANSEN', 'Method', 'Pred-accuracy'])
t.add_row(['MorganUnhashed', 'SVM', '0.823'])
t.add_row(['physchem', 'SVM', '0.661'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.689'])
print(t)
'''

'''
#plot results
plt.rcdefaults()
objects = ('SVM on MorganUnhashed', 'SVM on Morgan512', 'SVM on Physchem',
           'SVM+ on Physchem+Morgan512', 'SVM+ on Physchem+MorganUnhashed')
y_pos = np.arange(len(objects))
performance = [0.850, 0.813, 0.674, 0.702, 0.702]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Prediction Accuracy')
plt.title('Results on Bursi dataset')
plt.show()


plt.rcdefaults()
objects = ('SVM on MorganUnhashed', 'SVM on Morgan512', 'SVM on Physchem',
           'SVM+ on Physchem+Morgan512', 'SVM+ on Physchem+MorganUnhashed')
y_pos = np.arange(len(objects))
performance = [ 0.904, 0.869, 0.838, 0.849, 0.849]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Prediction Accuracy')
plt.title('Results on MMP dataset')
plt.show()

plt.rcdefaults()
objects = ('SVM on MorganUnhashed', 'SVM on Morgan512', 'SVM on Physchem',
           'SVM+ on Physchem+Morgan512', 'SVM+ on Physchem+MorganUnhashed')
y_pos = np.arange(len(objects))
performance = [ 0.823, 0.766, 0.639, 0.659, 0.686]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Prediction Accuracy')
plt.title('Results on HANSEN dataset')
plt.show()
'''
'''
t.clear_rows()
t.add_row(['MMP - desc', 'SVM', '0.845'])
t.add_row(['MMP - 512bitsUnhashed', 'SVM', '0.904'])
t.add_row(['MMP - desc+512bitsUnhashed', 'SVM+', '0.845'])
print(t)
t.clear_rows()
t.add_row(['HANSEN - desc', 'SVM', '0.845'])
t.add_row(['HANSEN - 512bitsUnhashed', 'SVM', '0.904'])
t.add_row(['HANSEN - desc+512bitsUnhashed', 'SVM+', '0.845'])
print(t)
'''

'''
#param C = 100.000000, gamma = 0.100000, pred accuracy = 0.867000
t = PrettyTable(['SVMFile', 'SVMPlusFile', 'C', 'Gamma', 'Pred-accuracy'])
t.add_row(['MMP - desc', 'MMP - 512bitsUnhashed', '100', '.1', '0.867'])
print(t)
'''

'''
#param C = 100.000000, gamma = 0.100000, pred accuracy = 0.706000
t = PrettyTable(['SVMFile', 'SVMPlusFile', 'C', 'Gamma', 'Pred-accuracy'])
t.add_row(['Bursi - Phys-chem', 'Bursi - morgan-unhashed', '100', '0.1', '0.706'])
print(t)

#param C = 100.000000, gamma = 0.100000, pred accuracy = 0.680000
t = PrettyTable(['File Name', 'C', 'Gamma', 'Pred-accuracy'])
t.add_row(['Bursi - desc', '100', '0.1', '0.68'])
t.add_row(['Bursi - 512bitsUnhashed', '10', '.01', '0.850', 'TBC'])
print(t)

param C = 10.000000, gamma = 0.100000, pred accuracy = 0.849000
t = PrettyTable(['File Name', 'C', 'Gamma', 'Pred-accuracy'])
t.add_row(['MMP - desc', '100', '0.1', '0.849'])
t.add_row(['MMP - 512bitsUnhashed', '10', '.01', '0.904', 'TBC'])
print(t)

param C = 100.000000, gamma = 0.100000, pred accuracy = 0.661000
t = PrettyTable(['File Name', 'C', 'Gamma', 'Pred-accuracy', 'AUC'])
t.add_row(['HANSEN - desc', '100', '0.1', '0.661', '0.661'])
t.add_row(['HANSEN - 512bitsUnhashed', '10', '.01', '0.823', 'TBC'])
print(t)
'''

'''
MMP
Average of SVM on X accuracy = 0.847522, svmXValidity = 0.164847, svmXOF = 0.260612
Average of SVM on XStar accuracy = 0.896726, svmXStartVal = 0.140734, svmXStartOF = 0.098628
Average of SVM Plus accuracy = 0.849292, svmPlusVal = 0.151952, svmPlusOF = 0.248843


Average of SVM on X accuracy = 0.664593, svmXValidity = 0.135885, svmXOF = 0.285180
Average of SVM on XStar accuracy = 0.804992, svmXStartVal = 0.123787, svmXStartOF = 0.128693
Average of SVM Plus accuracy = 0.664055, svmPlusVal = 0.137317, svmPlusOF = 0.283540

Average of SVM on X accuracy = 0.888889, svmXValidity = 0.168761, svmXOF = 0.272146
Average of SVM on XStar accuracy = 0.917857, svmXStartVal = 0.107204, svmXStartOF = 0.092754
Average of SVM Plus accuracy = 0.888889, svmPlusVal = 0.100159, svmPlusOF = 0.226047
'''