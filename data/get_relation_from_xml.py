import re
import os
from bs4 import BeautifulSoup


def get_relation(path):
    labels = os.listdir(path)
    f1 = open("relation.csv", 'w', encoding='utf-8')
    f1.writelines("\"{}\"\t\"{}\"\t\"{}\"".format("event1", "event2", "reltype"))
    f1.writelines('\n')
    Relation = re.compile('<erelation.*</erelation>')
    Eid = re.compile('eid=\"e\d*\"')
    Reltype = re.compile('reltype=\"\w*\"')
    count, success, wrong, manual = 0, 0, 0, 0
    for label in labels:
        files = os.listdir(path + '/' + label)
        for file in files:
            count += 1
            f = open(path + '/' + label + '/' + file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            body = str(soup.find_all('body')[0])
            relations = Relation.findall(body)
            body = body.replace('\n', '')
            for relation in relations:
                success += 1
                eid = Eid.findall(relation)
                reltype = str(Reltype.findall(relation)[0])
                reltype = re.sub(u"reltype=\"", "", reltype)
                reltype = re.sub(u"\"", "", reltype)
                try:
                    str1 = '<event '+eid[0]+'.*</event>'
                    Event1 = re.compile(str1)
                    event1 = Event1.findall(body)[0]
                    str2 = '<event ' + eid[1] + '.*</event>'
                    Event2 = re.compile(str2)
                    event2 = Event2.findall(body)[0]
                except:
                    wrong += 1
                    if reltype != 'Thoughtcontent':
                        manual += 1
                        print(body)
                        print(relation)
                event1 = re.sub(u"<.*?>", "", event1)
                event1 = event1.replace('\t', '')
                event2 = re.sub(u"<.*?>", "", event2)
                event2 = event2.replace('\t', '')
                f1.writelines("\"{}\"\t\"{}\"\t\"{}\"".format(event1, event2, reltype))
                f1.writelines('\n')
    f1.close()
    print("文件数：{}，关系数：{}，标注失败：{}, 需手动标注：{}, 其余关系为Thoughtcontent".format(count, success, wrong, manual))
    return


if __name__ == "__main__":
    data_path = "data/CEC"
    get_relation(data_path)
