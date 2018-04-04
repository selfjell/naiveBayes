from bayes import Bayes
import preprocess as pp

print("_"*15 + "BAYES CLASSIFIER" + "_"*15)
print(" "*20 + "MENU:")
print("TRAIN:\t\t T")
print("LOAD TRAINING:\t L")
print("SAVE TRAINING:\t S")
print("DISPLAY STATS:\t D")
print("CLASSIFY:\t C %FILEPATH%")
print("QUIT:\t\t Q")

while(True):
    command = input("Enter command:")
    command = command.lower()
    if command.startswith('t'):
        classifier = pp.main()
    elif command.startswith('l'):
        pass
    elif command.startswith('s'):
        pass
    elif command.startswith('c'):
        pass
    elif command.startswith('d'):
        scores = pp.load_stats()
        pp.print_stats(scores)
    elif command.startswith('q'):
        comf = input("Confirm Y/N:")
        if(comf.lower().startswith('y')):
            break
