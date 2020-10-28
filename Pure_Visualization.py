import matplotlib.pyplot as plt

# '''
plt.plot(126, 92.4, 'o', label="ICU")
plt.plot(187, 92.2, 'o', label="Model 1 (Grid Search)")
plt.plot(57, 91.9, 'o', label="Model 2")
plt.plot(38, 91.6, 'o', label="Model 3")
plt.plot(54, 92, 'o', label="Model 4 (Bayes optimization)")
plt.plot(54, 94.2, 'o', label="Model 4 with heavy training")
plt.plot(54, 92.7, 'o', label="Model 4 with heavy training (pseudo)")
plt.plot(19, 90.7, 'o', label="Model 5")
plt.ylim([90, 94.5])
plt.xlabel("Memory (KB)")
plt.ylabel("BIES Accuracy")

plt.legend(loc='lower right')
plt.show()
# '''

'''
input = [50, 100, 300, 500, 700, 1000]
output = [83.9, 86.3, 89, 89.6, 89.4, 89.4]
plt.xlabel("graph. clust. size")
plt.ylabel("BIES Accuracy")
plt.plot(input, output, marker="o", color="r")
plt.show()
'''

'''
input = [0.0001, 0.001, 0.01, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
output = [81, 88.3, 90.5, 91, 91.3, 91.3, 91.1, 89.1, 85.9]
plt.xlabel("learning rate")
plt.ylabel("BIES Accuracy")
plt.plot(input, output, marker="o")
plt.show()
'''

'''
input = [4, 8, 16, 20, 30, 40, 50, 64]
output = [88.5, 89.1 ,90.2 ,91.4, 91.6, 91.7, 91.1, 90.5]
plt.xlabel("hunits")
plt.ylabel("BIES Accuracy")
plt.plot(input, output, marker="o", color="g")
plt.show()
'''