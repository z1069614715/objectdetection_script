import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_list = ['a/face_Box.csv', 'b/face_Box.csv']
    names = ['improve', 'baseline']
    ap = ['0.673', '0.639']
    
    plt.figure(figsize=(6, 6))
    for i in range(len(file_list)):
        pr_data = pd.read_csv(file_list[i], header=None)
        recall, precision = np.array(pr_data[0]), np.array(pr_data[1])
        
        plt.plot(recall, precision, label=f'{names[i]} ap:{ap[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pr.png')