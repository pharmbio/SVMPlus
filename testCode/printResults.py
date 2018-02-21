from prettytable import PrettyTable
t = PrettyTable(['Dataset-BURSI', 'Method', 'Pred-accuracy'])
t.add_row(['physchem', 'SVM', '0.674'])
t.add_row(['Morgan512', 'SVM', '0.813'])
t.add_row(['MorganUnhashed', 'SVM', '0.850'])
t.add_row(['physchem+Morgan512', 'SVM+', '0.702'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.702'])
print(t)



t = PrettyTable(['Dataset-MMP', 'Method', 'Pred-accuracy'])
t.add_row(['physchem', 'SVM', '0.845'])
t.add_row(['Morgan512', 'SVM', '0.869'])
t.add_row(['MorganUnhashed', 'SVM', '0.904'])
t.add_row(['physchem+Morgan512', 'SVM+', '0.849'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.849'])
print(t)

t = PrettyTable(['Dataset-MMP', 'Method', 'Pred-accuracy'])
t.add_row(['physchem', 'SVM', '0.845'])
t.add_row(['Morgan512', 'SVM', '0.869'])
t.add_row(['MorganUnhashed', 'SVM', '0.904'])
t.add_row(['physchem+Morgan512', 'SVM+', '0.702'])
t.add_row(['physchem+MorganUnhashed', 'SVM+', '0.702'])
print(t)

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
t.add_row(['MMP - desc', '100', '0.001', '0.845', '0.747'])
t.add_row(['MMP - 64bits', '100', '0.1', '0.867', '0.801'])
t.add_row(['MMP - 512bits', '100', '0.0001', '0.869', '0.852'])
t.add_row(['MMP - 512bitsUnhashed', '10', '.01', '0.904', 'TBC'])
print(t)


t = PrettyTable(['File Name', 'C', 'Gamma', 'Pred-accuracy', 'AUC'])
t.add_row(['CAS - desc', '1000', '0.0001', '0.693', '0.748'])
t.add_row(['CAS - 64bits', '1', '0.1', '0.742', '0.817'])
t.add_row(['CAS - 512bits', '10', '.01', '0.771', 'TBC'])
t.add_row(['CAS - 512bitsUnhashed', '10', '.01', '0.823', 'TBC'])
print(t)
'''