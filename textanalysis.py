f= open("dwight.txt","w+")
with open("office.txt", "r") as ins:
    for line in ins: 
        if "Dwight:" in line: 
            f.write(line[8:(len(line))]) 
f.close() 