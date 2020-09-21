import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import sys

def get_candidates(k, frequents):
    # j gives length of first items in the candidate sets to check for correspondance to combine and generate next candidates
    j = k-1
    
    candidates = []    
    if k > 1:
        # check each frequent against all the others
        for f1 in range(len(frequents)):
            for f2 in range(f1+1, len(frequents)):
                # if first j items match of the current frequent set combine to get a next potential candidate
                if frequents[f1][0:j]==frequents[f2][0:j]:
                    candidates.append(frequents[f1][0:j+1]+frequents[f2][j:])
    # if k=1 combining is straight forward
    else:
        for f1 in range(len(frequents)):
            for f2 in range(f1+1, len(frequents)):
                candidates.append([frequents[f1],frequents[f2]])
    
    return candidates

def frequent_item_sets(adjacency_matrix, C1_itemset, min_sup):
    """
    Generate frequent item sets from the Pandas DataFrame. DataFrame indices are the transaction numbers and the DataFrame columns are the department IDs
    """
    # dictionary of itemsets and corresponding support
    support_dict = {}
    binary_matrix = adjacency_matrix.copy()
    # biggest possible itemset is all items
    max_itemset = len(C1_itemset)
    # first set of candidates is just a single itenset of each item
    candidates = C1_itemset
    
    frequents = []
    supports = []
    for candidate in candidates:
        support = binary_matrix[candidate].sum()
        
        # if min support is met, add candidate and support to lists
        if support >= min_sup:
                frequents.append(candidate)
                supports.append(support)

    if len(frequents) == 0:
        return support_dict

    support_dict[1] = [frequents, supports]

    # generate the next potential candidates
    candidates = get_candidates(1, frequents)

    # work through all potential lengths of itemsets and check if it meets minimum support   
    for k in range(2, max_itemset):
        # reset the list of frequet items and corresponding support for each round
        frequents = []
        supports = []
        # check support of each candidate
        for candidate in candidates:
            # calculates support for the candidate using built-in pandas dataframe operations
            support = binary_matrix.where(binary_matrix[candidate].sum(axis=1) == len(candidate)).sum().max()
            # if min support is met, add candidate and support to lists
            if support >= min_sup:
                frequents.append(candidate)
                supports.append(support)

        # if no itemsets meet minimum support requirement exit
        if len(frequents) == 0:
            print('No Frequents')
            return support_dict
        
        # add frequent item list and support list to the support dictionary with the key being the item set size
        support_dict[k] = [frequents, supports]        
        
        # generate the next potential candidates from current frequent item sets
        candidates = get_candidates(k, frequents)
        # prune the candidates if previous combinations of the items didn't meet the min support
        candidates = prune_candidates(candidates, frequents)

    return support_dict

def prune_candidates(candidates, frequent_set):
    pruned_list = []
    
    # check all candidates
    for c in range(len(candidates)):
        prune_flag = False
        # check each of the possible proceeding subsets of the candidate to remove the candidate if one of them is not part of the frequent itemset. Only need to check the level prior
        for i in range(len(candidates[c])):
            # resets the temp for each round
            temp = copy.deepcopy(candidates[c])
            # remove different item each round to make the subsets
            temp.pop(i)
            # if one of subsets not in the frequent itemset, don't add the candidate to the list and break out for loop cause other subsets do not need to be checked
            if temp not in frequent_set:
                prune_flag = True
                break
        # if all subsets in frequent itemsets add the candidate to the pruned list
        if not prune_flag:
            pruned_list.append(candidates[c])

    return pruned_list

def get_antecedents(parent_set, frequency_dict, max_key):
    antecedents = []
    # go through frequent itemsets to get the potential antecedents
    for key in frequency_dict.keys():
        # antecedents are subsets of the parent so they will be less than the max key which equlas size of the parent, key is equivalent to itemset length
        if key < max_key:
            # get all itemsets of length key
            f_set = frequency_dict[key]
            # iterate though the potential itemsets
            for i in range(len(f_set[0])):
                # this handles itemset list of size 1
                if len(list(f_set[0][i]))==1:
                    temp = [f_set[0][i]]
                else:
                    temp = f_set[0][i]
                # if the itemset is a subset of the parent it is an antecedent. Add itemset and support to antecedent list
                if set(temp).issubset(set(parent_set)):
                    antecedents.append([temp, f_set[1][i]])
                
        else:
            break
       
    return antecedents
                
def get_association_rules(frequency_dict, min_conf):
    confidence_dict = []
    # iterate through the different itemset sizes
    for key in frequency_dict.keys():
        if key > 1:
            # itereate through the differnt frequent items
            for i in range(len(frequency_dict[key][0])):
                # confidence = sup|Z|/sup|X|
                z = frequency_dict[key][0][i]
                z_sup = frequency_dict[key][1][i]

                # get antecedents (all proceeding subsets of z)
                antecedents = get_antecedents(z, frequency_dict, key)

                # calculate confidence of Z to X
                while len(antecedents) > 0:
                    # check maximal antecedent
                    antecedent = antecedents.pop()
                    x = set(antecedent[0])
                    x_sup = antecedent[1]
                    y = set(z)-x

                    confidence = z_sup/x_sup
                    # if confidence greater than the minimum add it to the dictionary else prune all antecedents that are subsets of x as they will not meet the minumu confidence either
                    if confidence >= min_conf:
                        confidence_dict.append([list(x),list(y), confidence])
                    else:
                        antecedents = prune_antecedents(x, antecedents)
                            
    return confidence_dict

def prune_antecedents(x, antecedents):
    pruned_antecedents = []
    # iterate through all antecedents
    for antecedent in antecedents:
        # if antecedent is not a subset of x then add it to the pruned list
        if set(antecedent[0]).issubset(x):
            pass
        else:
            pruned_antecedents.append(antecedent)
            
    return pruned_antecedents

def get_evals(confidences_list, frequency_dict, size_dataset):
    metrics = []
    # go through itemsets that met the minumum confidence requirement
    for c in confidences_list:
        # c = [X, Y, confidence_XY]
        # lift = confidence_XY / relSupport(Y)
        # confidence value stored in the 2 array spot
        confidence = c[2]
        # find length of Y
        k = len(c[1])
        frequencies = frequency_dict[k]
        # find Y
        for i in range(len(frequencies[0])):
            # handles itemset of size 1
            if k==1:
                temp = set([frequencies[0][i]])
            else:
                temp = set(frequencies[0][i])
            # if true Y is found
            if set(c[1])==temp:
                support = frequencies[1][i]
        # calculate lift
        lift = confidence/(support/size_dataset)
        
        # append lift to c
        c.append(lift)
        
        ## Calculate leverage
        # leverage = relSup(XY) - relSup(X)*relSup(Y)

        x = set(c[0])
        y = set(c[1])
        xy = x.union(y)

        # find length XY
        k = len(xy)
        frequencies = frequency_dict[k]
        # find XY
        for i in range(len(frequencies[0])):
            # handles itemset of size 1
            if k==1:
                temp = set([frequencies[0][i]])
            else:
                temp = set(frequencies[0][i])
            if xy==temp:
                support_xy = frequencies[1][i]
        # calculate rel support XY
        rel_xy = support_xy/size_dataset
        
        # find length X
        k = len(x)
        frequencies = frequency_dict[k]
        # find X
        for i in range(len(frequencies[0])):
            # handles itemset of size 1
            if k==1:
                temp = set([frequencies[0][i]])
            else:
                temp = set(frequencies[0][i])
            if x==temp:
                support_x = frequencies[1][i]
        # calculate rel support X
        rel_x = support_x/size_dataset
        
        # find length Y
        k = len(y)
        frequencies = frequency_dict[k]
        # find Y
        for i in range(len(frequencies[0])):
            # handles itemset of size 1
            if k==1:
                temp = set([frequencies[0][i]])
            else:
                temp = set(frequencies[0][i])
            if y==temp:
                support_y = frequencies[1][i]
        # calculate rel support Y
        rel_y = support_y/size_dataset
        
        # calculate leverage
        leverage = rel_xy - rel_x*rel_y

        # append leverage to c
        c.append(leverage)

        # save seperate array of just lifts and leverage
        metrics.append([lift, leverage])
        
    # Confidences_list = [X, Y, confidence, lift, leverage]
    # metrics = [lift, leverage]
    return confidences_list, metrics

if __name__ == "__main__":
        data = pd.read_csv(sys.argv[1])
        minSuport = sys.argv[2]
        conf = sys.argv[3]
        numberOfRules = sys.argv[4]
        depts = set(data['Dept'])
        transactions = set(data['POS Txn'])
        data["ID"] = data["ID"].astype(str)
        ids = set(data["ID"])
        
        adjacency_matrix = pd.DataFrame(columns = ids, index = transactions, data = np.zeros((len(transactions), len(ids))))
        
        for row in data.iterrows():
            adjacency_matrix[row[1][2]].loc[row[1][0]] += 1

        # frequency = { key: [ list([frequent itemsets]), list([itemsets support])]}
        frequency = frequent_item_sets(adjacency_matrix, adjacency_matrix.columns, int(minSuport))

        
        # confidences = [X, Y, confidence]
        confidences = get_association_rules(frequency, float(conf))


        # evals = [X, Y, confidence, lift, leverage]
        # metrics = [lift, leverage]
        evals, metrics = get_evals(confidences, frequency, len(adjacency_matrix))
        

        sport = sorted(evals, key=lambda evals: evals[3], reverse=True)

        for row in sport[0:int(numberOfRules)]:
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
