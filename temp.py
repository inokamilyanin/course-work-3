import os
files = os.listdir("Проект/")
print(f"Number of files in Проект directory: {len(files)}")
print("\nFiles list:")
for file in files:
    print(file)