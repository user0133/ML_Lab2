import string
# Task 10.11
def find_reverse_pairs(word_list):
    reverse_pairs = []
    word_set = set(word_list)  # Convert the word list to a set for O(1) lookups

    for word in word_list:
        reversed_word = word[::-1]
        if reversed_word in word_set and reversed_word != word:
            reverse_pairs.append((word, reversed_word))
            word_set.remove(word)
            word_set.remove(reversed_word)

    return reverse_pairs
words = ['hello','bye', 'boon','noob', 'star', 'rats']
reverse_pairs = find_reverse_pairs(words)
print("Reverse pairs:", reverse_pairs)

# Task 11.4
def has_duplicates(numbers):
    freq_dict = {}
    for num in numbers:
        if num in freq_dict:
            return True
        freq_dict[num] = 1  # Add item to dictionary if not already present
    return False


# Example usage
numbers = [1, 2, 3, 4, 5, 3]
print(has_duplicates(numbers))

numbers_no_duplicates = [1, 2, 3, 4, 5]
print(has_duplicates(numbers_no_duplicates))

# Task 13.1
def process_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            words = line.strip().split()
            #remove punctuation, whitespace, and convert to lowercase
            formatted_words = [format_word(word) for word in words]
            print(formatted_words)

def format_word(word):
    return word.strip(string.whitespace + string.punctuation).lower()

filename = 'text_file.txt'
process_file(filename)

