
# Cleans a list of String-reviews to lower(), no duplicate words, and negation fix for sentiment analysis
# Argument input_text = A list of String-reviews
# Return input_text = The cleaned up list for sentiment analysis
# Made by Jakob
def clean_text(input_text):
    for i in range(len(input_text)):

        # String to lowercase letters and removes the <br /> thing
        input_text[i] = input_text[i].lower().replace("\n", " ").replace("<br />", " ").replace("  ", " ")

        # Convert the the review in the form of a String into a list of words and removes the duplicate words
        words = []
        [words.append(x) for x in input_text[i].split(" ") if x not in words]

        # Ads NOT_ in front of the words following a negation operator: "not", "n't", "no" and "never"
        negation_word = ""
        for j in range(len(words)):
            words[j] = negation_word + words[j]
            if words[j] == "not" or words[j][-3:] == "n't" or words[j] == "no" or words[j] == "never":
                negation_word = "NOT_"
            if words[j][-1] == "." or words[j][-1] == "!" or words[j][-1] == "?" or words[j][-1] == ",":
                words[j] = words[j].replace(words[j][-1], "")
                negation_word = ""

        # Joins the list "words" back into String-format and puts it back into its place
        input_text[i] = ' '.join(words)

    # Returns a list of Strings with all the changes made
    return input_text










# ---------
# TESTING OF THE ABOVE
paths = []
paths.append("../DATA/aclImdb/train/neg/1_1.txt")
paths.append("../DATA/aclImdb/train/neg/2_1.txt")
paths.append("../DATA/aclImdb/train/neg/3_4.txt")
paths.append("../DATA/aclImdb/train/neg/5_3.txt")

# A list of String-reviews in order to test clean_text() with some sample reviews
# Is expanded for each make_text_list() run
text = []


def make_text_list(input_path):
    with open(input_path) as f:
        line = f.read()
        text.append(line)
        print(line)

# Before clean_text()
print("Input samples: ")
for path in paths:
    make_text_list(path)

# After clean_text()
print("\nOutput from samples")
new_text = clean_text(text)
for review in new_text:
    print(review)