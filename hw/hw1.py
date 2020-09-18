import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import sys

def get_candidates(k, frequents):
    j = k-1
    candidates = []    
    if k > 1:
#             print('here ', k)
            for f1 in range(len(frequents)):
                for f2 in range(f1+1, len(frequents)):
                    if frequents[f1][0:j]==frequents[f2][0:j]:
                        candidates.append(frequents[f1][0:j+1]+frequents[f2][j:])
    else:
        for f1 in range(len(frequents)):
            for f2 in range(f1+1, len(frequents)):
                candidates.append([frequents[f1],frequents[f2]])
    
    return candidates

def frequent_item_sets(adj_matrix, C1_itemset, min_sup):
    support_dict = {}
    binary_matrix = adj_matrix.copy()
    max_itemset = len(C1_itemset)
    candidates = C1_itemset
    
    frequents = []
    supports = []
    for candidate in candidates:
        support = binary_matrix[candidate].sum()
        

        if support >= min_sup:
                frequents.append(candidate)
                supports.append(support)
#                 print(candidate, support)

    if len(frequents) == 0:
        return support_dict

    support_dict[1] = [frequents, supports]
    candidates = get_candidates(1, frequents)
                    
    for k in range(2, max_itemset):
        print('Frequency set size: ', k)
        frequents = []
        supports = []
        for candidate in candidates:
            support = binary_matrix.where(binary_matrix[candidate].sum(axis=1) == len(candidate)).sum().max()
            if support >= min_sup:
                frequents.append(candidate)
                supports.append(support)
#                 print(candidate, support)

                
        if len(frequents) == 0:
            print('No Frequents')
            return support_dict
        
        support_dict[k] = [frequents, supports]
#         print(frequents)
        
        
        candidates = get_candidates(k, frequents)
#         print('full candidates ', candidates)
        candidates = prune_candidates(candidates, frequents)
#         print('pruned ', candidates)

    return support_dict

def prune_candidates(candidates, frequent_set):
    pruned_list = []
    
    for c in range(len(candidates)):
        prune_flag = False
        for i in range(len(candidates[c])):
            temp = copy.deepcopy(candidates[c])
            temp.pop(i)
            if temp not in frequent_set:
                prune_flag = True
                break
        if not prune_flag:
            pruned_list.append(candidates[c])
    return pruned_list

def get_antecedents(parent_set, frequency_dict, max_key):
    antecedents = []
    for key in frequency_dict.keys():
        if key < max_key:
            f_set = frequency_dict[key]
            print('f_set', list(f_set[0][1]))
#             print('f_set_can', f_set[0])
#             print('f_set_sup', f_set[1])
            for i in range(len(f_set[0])):
                print(f_set[0][i])
                if len(list(f_set[0][i]))==1:
                    temp = [f_set[0][i]]
                else:
                    temp = f_set[0][i]
#                 print(set(temp))
#                 print(set(parent_set))
                if set(temp).issubset(set(parent_set)):
#                     print(temp,f_set[1][i])
                    antecedents.append([temp,f_set[1][i]])
                
        else:
            break
                                        
                    
    return antecedents
                
def get_association_rules(frequency_dict, min_conf):
    confidence_dict = []
    for key in frequency_dict.keys():
        if key > 1:
            for i in range(len(frequency_dict[key][0])):
                z = frequency_dict[key][0][i]
                z_sup = frequency_dict[key][1][i]
                print('parent set: ', z, z_sup)
                antecedents = get_antecedents(z, frequency_dict, key)

                while len(antecedents) > 0:
                    antecedent = antecedents.pop()
                    x = set(antecedent[0])
                    x_sup = antecedent[1]
                    print('X', x, x_sup)
                    y = set(z)-x

                    confidence = z_sup/x_sup
                    print(x, '->', y, confidence)
                    if confidence >= min_conf:
                        confidence_dict.append([list(x),list(y), confidence])
                    else:
                        antecedents = prune_antecedents(x, antecedents)
                            
    return confidence_dict

def prune_antecedents(x, antecedents):
    pruned_antecedents = []
    for antecedent in antecedents:
        if set(antecedent[0]).issubset(x):
            pass
        else:
            pruned_antecedents.append(antecedent)
            
    return pruned_antecedents

def get_evals(confidences_list, frequency_dict, size_dataset):
    metrics = []
    for c in confidences_list:
        print('\n', c)
        confidence = c[2]
        k = len(c[1])
        frequencies = frequency_dict[k]

        for i in range(len(frequencies[0])):
            if k==1:
                temp = set([frequencies[0][i]])
            else:
                temp = set(frequencies[0][i])
            if set(c[1])==temp:
                support = frequencies[1][i]
        lift = confidence/(support/size_dataset)
        print('lift', lift)
        
        c.append(lift)
        
        x = set(c[0])
        y = set(c[1])
        xy = x.union(y)

        k = len(xy)
        frequencies = frequency_dict[k]
        for i in range(len(frequencies[0])):
            if k==1:
                temp = set([frequencies[0][i]])
            else:
                temp = set(frequencies[0][i])
            if xy==temp:
                support_xy = frequencies[1][i]
        rel_xy = support_xy/size_dataset
        
        x = set(c[0])
        k = len(x)
        frequencies = frequency_dict[k]
        for i in range(len(frequencies[0])):
            if k==1:
                temp = set([frequencies[0][i]])
            else:
                temp = set(frequencies[0][i])
            if x==temp:
                support_x = frequencies[1][i]
        rel_x = support_x/size_dataset
        
        y = set(c[1])
        k = len(y)
        frequencies = frequency_dict[k]
        for i in range(len(frequencies[0])):
            if k==1:
                temp = set([frequencies[0][i]])
            else:
                temp = set(frequencies[0][i])
            if y==temp:
                support_y = frequencies[1][i]
        rel_y = support_y/size_dataset
        
        leverage = rel_xy - rel_x*rel_y
        print('leverage', leverage)
        
        c.append(leverage)
        metrics.append([lift, leverage])
        
    return confidences_list, metrics

if __name__ == "__main__":
        data = pd.read_csv(sys.argv[1])
        depts = set(data['Dept'])
        transactions = set(data['POS Txn'])
        ids = set(data["ID"])
        
        adj_matrix = pd.DataFrame(columns = ids, index = transactions, data = np.zeros((len(transactions), len(ids))))
        
        for row in data.iterrows():
            adj_matrix[row[1][2]].loc[row[1][0]] += 1

        frequency = frequent_item_sets(adj_matrix, adj_matrix.columns, 3)
        for key in frequency.keys():
            print('Item Set Length: ', key, '\n', frequency[key])
        confidences = get_association_rules(frequency, 1)
        for c in confidences:
            print(c)
        evals, metrics = get_evals(confidence_full, frequent_full, len(adj_matrix))
        for row in evals_full:
            print(row[0], '-->', row[1], '\nConfidence: ', row[2], '\nLift: ', row[3], '\nLeverage: ', row[4])
        for row in m:
            print(row)
        


        


        # test2 = [
        # [1, 1, 1, 0, 0],
        # [1, 1, 1, 1, 1],
        # [1, 0, 1, 1, 0],
        # [1, 0, 1, 1, 1],
        # [1, 1, 1, 1, 0]
        # ]

        # test_df = pd.DataFrame(data=test2, columns = ['A', 'B', 'C', 'D', 'E'])
        # frequent_test = frequent_item_sets(test_df, test_df.columns, 2)
        # print('\n')
        # for key in frequent_test.keys():
        #     print('Item Set Length: ', key, '\n', frequent_test[key][0])
        # confidence = get_association_rules(frequent_test, 1)
        # for c in confidence:
        #     print(c)

        # get_lifts(confidence, frequent_test, len(test_df)) 