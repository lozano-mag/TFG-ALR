import re

def extract_info_from_txt(filename):
    with open(filename, 'r' , encoding='utf-8') as file:
        content = file.read()

    # Create a list to store the blocks of words
    blocks = []

    # Find all the titles
    titles = re.findall(r"'title': '([^']*)'", content)
    for title in titles:
        blocks.append(title.split())

    # Find all the descriptions
    descriptions = re.findall(r"'description': '([^']*)'", content)
    for description in descriptions:
        blocks.append(description.split())

    return blocks

def write_info_to_txt(filename, blocks):
    with open(filename, 'w') as file:
        for block in blocks:
            # Check if the block is not empty
            if block:
                for word in block:
                    file.write(f"{word} X\n")
                file.write(". X\n\n")

def main():
    blocks = extract_info_from_txt('data_stix.txt')
    write_info_to_txt('result.txt', blocks)

if __name__ == "__main__":
    main()
