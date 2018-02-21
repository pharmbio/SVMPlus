from prettytable import PrettyTable
t = PrettyTable(['SVMFile', 'SVMPlusFile', 'C', 'Gamma', 'Pred-accuracy'])
t.add_row(['MMP - desc', 'MMP - 512bitsUnhashed', '10', '.001', '0.845'])
print(t)


'''
from prettytable import PrettyTable
t = PrettyTable(['SVMFile', 'SVMPlusFile', 'C', 'Gamma', 'Pred-accuracy'])
t.add_row(['Bursi - desc', 'Bursi - 512bitsUnhashed', '0.0001', '0.674', '0.742'])
print(t)

from prettytable import PrettyTable
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