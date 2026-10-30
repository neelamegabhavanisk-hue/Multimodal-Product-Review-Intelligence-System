import zipfile
import xml.etree.ElementTree as ET

path = r'C:\Users\Neela Mega Bhavani\Final Project\Multimodal Product Review Intelligence System.docx'
with zipfile.ZipFile(path) as z:
    root = ET.fromstring(z.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paras = []
    for p in root.findall('.//w:p', ns):
        text = ''.join(t.text or '' for t in p.findall('.//w:t', ns))
        paras.append(text)

print('\n'.join(paras[:250]))
