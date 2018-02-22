from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


t = PrettyTable(['Dataset-BURSI', 'Method', 'Pred-accuracy'])
t.add_row(['Morgan512', 'SVM', '0.813'])
t.add_row(['MorganUnhashed', 'SVM', '0.850'])
t.add_row(['physchem', 'SVM', '0.674'])
t.add_row(['physchem+Morgan512', 'SVM+', '0.702'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.702'])
print(t)

t = PrettyTable(['Dataset-MMP', 'Method', 'Pred-accuracy'])
t.add_row(['Morgan512', 'SVM', '0.869'])
t.add_row(['MorganUnhashed', 'SVM', '0.904'])
t.add_row(['physchem', 'SVM', '0.838'])
t.add_row(['physchem+Morgan512', 'SVM+', '0.849'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.849'])
print(t)

t = PrettyTable(['Dataset-CAS', 'Method', 'Pred-accuracy'])
t.add_row(['Morgan512', 'SVM', '0.766'])
t.add_row(['MorganUnhashed', 'SVM', '0.823'])
t.add_row(['physchem', 'SVM', '0.639'])
t.add_row(['physchem+Morgan512', 'SVM+', '0.659'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.686'])
print(t)


'''
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
t = PrettyTable(['SVMFile', 'SVMPlusFile', 'C', 'Gamma', 'Pred-accuracy'])
t.add_row(['MMP - desc', 'MMP - 512bitsUnhashed', '10', '.001', '0.845'])
print(t)
'''

'''
t = PrettyTable(['SVMFile', 'SVMPlusFile', 'C', 'Gamma', 'Pred-accuracy'])
t.add_row(['Bursi - desc', 'Bursi - 512bitsUnhashed', '0.0001', '0.674', '0.742'])
print(t)

t = PrettyTable(['File Name', 'C', 'Gamma', 'Pred-accuracy', 'AUC'])
t.add_row(['Bursi - desc', '100', '0.0001', '0.674', '0.742'])
t.add_row(['Bursi - 64bits', '1', '0.1', '0.796', '0.861'])
t.add_row(['Bursi - 512bits', '1', '0.1', '0.813', '0.890'])
t.add_row(['Bursi - 512bitsUnhashed', '10', '.01', '0.850', 'TBC'])
print(t)


t = PrettyTable(['File Name', 'C', 'Gamma', 'Pred-accuracy', 'AUC'])
t.add_row(['MMP - desc', '100', '0.001', '0.838', '0.747'])
t.add_row(['MMP - 64bits', '100', '0.1', '0.867', '0.801'])
t.add_row(['MMP - 512bits', '100', '0.0001', '0.869', '0.852'])
t.add_row(['MMP - 512bitsUnhashed', '10', '.01', '0.904', 'TBC'])
print(t)


t = PrettyTable(['File Name', 'C', 'Gamma', 'Pred-accuracy', 'AUC'])
t.add_row(['CAS - desc', '100', '0.0001', '0.639', '0.748'])
t.add_row(['CAS - 64bits', '1', '0.1', '0.742', '0.817'])
t.add_row(['CAS - 512bits', '10', '.01', '0.771', 'TBC'])
t.add_row(['CAS - 512bitsUnhashed', '10', '.01', '0.823', 'TBC'])
print(t)
'''