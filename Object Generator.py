# bangla_stop_words = input("Enter Bangla Stop Words")

contents = []
while True:
    try:
        line = input()
    except EOFError:
        break
    contents.append(line)

print(contents)

bangla_stop_words = "{"
for index in contents:
    bangla_stop_words = bangla_stop_words + "'" + index + "',"

bangla_stop_words = bangla_stop_words + "}"

print(bangla_stop_words)

f = open("bangla_stop_words.txt", "w")
f.write(bangla_stop_words)
f.close()