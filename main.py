from bayes import Bayes
from pathlib import Path
import preprocess as pp

def print_menu():
    print(" "*20 + "MENU:")
    print("TRAIN:\t\t T")
    print("LOAD TRAINING:\t L")
    print("SAVE TRAINING:\t S")
    print("DISPLAY STATS:\t D")
    print("CLASSIFY:\t C %FILENAME%") #Should be in same directory
    print("REFRESH MENU:\t M")
    print("QUIT:\t\t Q")

print("_"*15 + "BAYES CLASSIFIER" + "_"*15)
print_menu()
classifier = None

while(True):
    command = input("Enter command:")
    command = command.lower()
    #Train clause
    if command.startswith('t'):
        classifier = pp.main()
    #Load training clause
    elif command.startswith('l'):
        print("Loading: ", end = '')
        if classifier is None:
            classifier = Bayes(trained = True)
        else:
            classifier.load()
    #Save training clause
    elif command.startswith('s'):
        print("Saving: ", end = '')
        if classifier is not None:
            classifier.save()
        else:
            print("Nothing to save")
    #Classify clause
    elif command.startswith('c'):
        if classifier is None:
            print("Load training first")
        else:
            path = command.split(" ")
            if(len(path)<2):
                print("Please enter filepath")
            else:
                path = path[1]
                path = Path('.').joinpath(path)
                text = ""
                try:
                    text = path.open('r',encoding = 'utf-8').read()
                except OSError as e:
                    print("File doesn't exist/Invalid path")
                print(text)
                text = pp.clean_text(text)
                pos,neg = classifier.test(text)
                print("CLASS: ", end = '')
                if pos == classifier.pos_prior or neg == classifier.neg_prior:
                    print("SOMETHING WENT WRONG")
                elif pos >= neg:
                    print("POSITIVE")
                else:
                    print("NEGATIVE")

    #Display stats clause
    elif command.startswith('d'):
        scores = pp.load_stats()
        pp.print_stats(scores)
    #Refresh menu clause
    elif command.startswith('m'):
        print_menu()
    #Quit clause
    elif command.startswith('q'):
        comf = input("Confirm Y/N:")
        if(comf.lower().startswith('y')):
            break
