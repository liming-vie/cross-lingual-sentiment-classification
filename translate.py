#!/usr/bin/env python
# encoding: utf-8

import re
import os
import urllib
from tqdm import tqdm
from mtranslate import translate
import xml.etree.ElementTree as ET

def len_url(text):
    return len(urllib.quote(text.encode('utf-8')))

def trans(text, target):
    try:
        text = translate(text.encode('utf-8'), target, 'auto')
    except:
        # manually translate by website
        text = "TODO"
    return text

def add_translate(parent, name, text, target):
    """
    Add translateed text into XML node parent
    """
    l=len_url(text)
    if l > 0:
        # deal with length limit
        if l > 1900:
            ps=text.split('\n')
            ret=[]
            for p in ps:
                l=len_url(p)
                if l>1900:
                    strs = p.split(u'。')
                    tmp=[]
                    for s in strs:
                        tmp.append(trans(s, target))
                    ret.append('. '.join(tmp))
                else:
                    ret.append(trans(p, target))
            text = '\n'.join(ret)
        else:
            text=trans(text, target)
    child = ET.SubElement(parent, name)
    child.text=text

def proc_text(node):
    if node.text:
        node.text=node.text.strip()
        return node.text
    return ""

if __name__ == '__main__':
    data_dir = '../eval/data'
    output_dir = '../output/eval/'
    domain_list = ['']#,['Unlabel_CN', 'Train_EN', 'Train_CN']
    target_list = ['en']#,'en', 'zh', 'en']
    file_list = ['dvd.xml']# ['book.xml', 'dvd.xml', 'music.xml']

    for domain, target in zip(domain_list, target_list):
        doutput = os.path.join(output_dir, domain)
        if not os.path.exists(doutput):
            os.makedirs(doutput)
        for file_name in file_list:
            file_input = os.path.join(data_dir, domain, file_name)
            print file_input
            file_output = os.path.join(doutput, file_name)
            text=open(file_input).read().decode('utf-8')
            # remove invalid characters
            text=re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+",u"",text)
            root = ET.fromstring(text.encode('utf-8'))
            with open(file_output, 'w') as fout:
                # use tqdm to show the progress bar
                for item in tqdm(root):
                    text=None
                    summary=None
                    for child in item:
                        if child.tag == 'text':
                            text=proc_text(child)
                        elif child.tag == 'summary':
                            summary = proc_text(child)
                    add_translate(item, 'tr_text', text, target)
                    add_translate(item, 'tr_summary', summary, target)
                tree = ET.ElementTree(root)
                tree.write(fout, encoding="UTF-8",xml_declaration=True)

