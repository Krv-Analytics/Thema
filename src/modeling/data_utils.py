
def convert_keys_to_alphabet(dictionary):
    base = 26  # Number of letters in the alphabet
    new_dict = {}

    keys = list(dictionary.keys())
    for i, key in enumerate(keys):
        # Calculate the position of each letter in the new key
        position = i
        new_key = ""
        while position >= 0:
            new_key = chr(ord('a') + (position % base)) + new_key
            position = (position // base) - 1

        new_dict[new_key] = dictionary[key]

    return new_dict