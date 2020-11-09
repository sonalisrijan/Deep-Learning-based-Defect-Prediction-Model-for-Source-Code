import pickle

def count_bugs(name_of_pickle_file):

    with open(name_of_pickle_file, "rb") as filename:
        y=pickle.load(filename)
    # NOTE: Based on whether you're calculating buggy rate for Training or Test/Validation data, you need to select one of the y_trim statements below 
    #y_trim = y[:50000] # For training data
    y_trim = y[:25000] # For Test/Validation data
    print (len(y_trim))
    print (type(y_trim), len(y_trim))
    bug_count=0
    for item in y_trim:
        if item==1:
            bug_count+=1
    print ("Filename is: %s" %(name_of_pickle_file))
    print ("Total number of buggy instances is: %d" %bug_count)
    print ("Total number of instances is: %d" %len(y_trim))
    print ("The buggy rate is: %.3f" %(bug_count/len(y_trim)))
    print ("---")

def main():
    #count_bugs("./y_train.pickle")
    count_bugs("./y_valid.pickle")
    count_bugs("./y_test.pickle")

main()

