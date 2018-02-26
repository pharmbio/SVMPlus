from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Comparing validiy and obs-fuzz using SVM and SVMPlus
t = PrettyTable(['Dataset-BURSI', 'M/L Method', 'Validity', 'Obs-fuzziness'])
t.add_row(['MorganUnhashed', 'SVM', '', ''])
t.add_row(['physchem', 'SVM', '.17', '.26'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '.12', '22'])
print(t)



'''
# SVM and SVM+ comparision with reduced dataset
t = PrettyTable(['Dataset-BURSI', 'Method', 'Pred-accuracy'])
t.add_row(['MorganUnhashed', 'SVM', '0.850'])
t.add_row(['physchem', 'SVM', '0.680'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.706'])
print(t)

t = PrettyTable(['Dataset-MMP', 'Method', 'Pred-accuracy'])
t.add_row(['MorganUnhashed', 'SVM', '0.904'])
t.add_row(['physchem', 'SVM', '0.849'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.867'])
print(t)

t = PrettyTable(['Dataset-CAS', 'Method', 'Pred-accuracy'])
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
plt.title('Results on CAS dataset')
plt.show()
'''
'''
t.clear_rows()
t.add_row(['MMP - desc', 'SVM', '0.845'])
t.add_row(['MMP - 512bitsUnhashed', 'SVM', '0.904'])
t.add_row(['MMP - desc+512bitsUnhashed', 'SVM+', '0.845'])
print(t)
t.clear_rows()
t.add_row(['CAS - desc', 'SVM', '0.845'])
t.add_row(['CAS - 512bitsUnhashed', 'SVM', '0.904'])
t.add_row(['CAS - desc+512bitsUnhashed', 'SVM+', '0.845'])
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
t.add_row(['CAS - desc', '100', '0.1', '0.661', '0.661'])
t.add_row(['CAS - 512bitsUnhashed', '10', '.01', '0.823', 'TBC'])
print(t)
'''