import matplotlib.pyplot as plt
import numpy as np

# Example data for the bar chart
categories = ['Hu\'s Moments', 'PHOG', 'Original']
values1 = [0.174, 0.5475, 0.6324]  # First group
values2 = [0.19334109669206337, 0.6087269055938698, 0.919]     # Second group to be bundled


x = np.arange(len(categories))
width = 0.3  # the width of the bars


color1 = ['green', 'green', 'purple']
plt.bar(x - width/2, values1, width, color=color1, edgecolor='black')


color2 = ['blue', 'blue', 'orange']
plt.bar(x + width/2, values2, width, color=color2, edgecolor='black')


plt.xticks(x, categories)

plt.grid(axis='y', linestyle='--', color='gray')
plt.ylim(0, 1)
plt.ylabel('Accuracy')

custom_legend = [plt.Rectangle((0, 0), 1, 1, color='green', edgecolor='black'),
                 plt.Rectangle((0, 0), 1, 1, color='blue', edgecolor='black'), 
                 plt.Rectangle((0, 0), 1, 1, color='purple', edgecolor='black'), 
                 plt.Rectangle((0, 0), 1, 1, color='orange', edgecolor='black')]


plt.legend(custom_legend, ['KNN', 'SVM', 'VGG16', 'YOLO11'], loc='upper left')

plt.show()
