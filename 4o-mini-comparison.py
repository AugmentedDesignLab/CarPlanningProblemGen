import matplotlib.pyplot as plt
import numpy as np

#correctness comparison for experiments using 1 scenario ID
shot_type = ["0", "2", "4", "6"]
#(52,53) has 17 interactions
range_52_53 = [14, 7, 8, 9]
#(6,7) has 7 interactions
range_6_7 = [0, 1, 3, 3]
x = np.arange(len(shot_type))
width = 0.5
plt.bar(x - width/2, range_52_53, width, label='(52,53)')
plt.bar(x + width/2, range_6_7, width, label='(6,7)')

plt.xticks(x, shot_type)
plt.xlabel('Shot Type')
plt.ylabel('# of Correctness Scores <= 5')
plt.title('Comparison Between 1 ID Experiments')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.show()


#NOTICE: BECAUSE THE DIFFERENT IDs HAVE DIFFERENT AMOUNTS OF CONTENT, BRING THEM DOWN TO THE SAME RATIO TO MAKE BETTER COMPARISONS
#RUN THIS PROGRAM TO BETTER UNDERSTAND