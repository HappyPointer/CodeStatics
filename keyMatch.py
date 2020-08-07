def keyMatch(article="",S=1):
    # print(S)
    with open(r".\data\medical_voc.txt","r",encoding='utf-8') as readFile:
        scan = article
        content = readFile.readline().replace("\n","").replace("\t","")
        result = []
        # print(scan)
        while(content):
            contents = content.split(" ")
            slength = len(contents)
            zlength = 0
            for c in contents:
                if c in scan:
                    zlength += 1  
            if zlength/slength >= S:
                result.append(content)
            content = readFile.readline().replace("\n","").replace("\t","")
        if len(result) > 5 :
            return result
        else:
            return keyMatch(article,S-0.02)

# with open(r"数据文件\\keyFile.txt","r",encoding='utf-8') as scanFile:
#     scan = ''.join(scanFile.readlines())
#     print(keyMatch(scan))

