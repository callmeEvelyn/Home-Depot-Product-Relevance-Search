import csv
import nltk
# nltk.download('wordnet')
import pandas as pd
import regex as re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
# from google_dict import google_dict
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
# from optparse import OptionParser
from time import time
from nltk.corpus import wordnet as wn
import difflib
import Levenshtein
import gensim
from gensim import models
from gensim.similarities import MatrixSimilarity
from gensim.models.tfidfmodel import TfidfModel
from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit,train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from scipy.stats import uniform, randint

'''
Functions for text preprocessing
'''

# convert to lower case
def toLowerCase(texts):
        return texts.lower()
stop_words = set(stopwords.words('english'))
# print(stop_words)

# read google_dict from file
def read_obj_from_pkl(fn):
    with open(fn,"rb") as f:
        obj=pickle.load(f)
    return obj
google_dict = read_obj_from_pkl('data/google_dict.pkl')

def remove_stop_word(texts):
    word_tokens = word_tokenize(texts)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return " ".join(filtered_sentence)
    
# helper class
def convert(texts, pattern_list):
    for pattern, replace in pattern_list:
        try:
            texts = re.sub(pattern, replace, texts)
        except:
            pass
    return re.sub(r'\s+', ' ', texts).strip()

# convert text number to digits
def textToDigits(texts):
    # convert text number to digits
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
           "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90]
    pattern_list = [(r"(?<=\W|^)%s(?=\W|$)"%n, str(d)) for n,d in zip(numbers, digits)]
    return convert(texts, pattern_list)
    
def parseHtml(texts):
        texts = BeautifulSoup(texts, "html.parser").get_text(separator=" ")
        return texts
    
# Removed commas between digits (for example, 10, 000 was replaced with 10000).
def removeDigitCommas(texts):
    pattern_list = [
            (r"(?<=\d+),(?=000)", r""),
        ]
    return convert(texts, pattern_list)

# remove special characters https://www.kaggle.com/steubk/fixing-typos/code
def removeSpecialCharacters(s):
    s = s.replace("craftsm,an", "craftsman")
    s = re.sub(r'depot.com/search=', '', s)
    s = re.sub(r'pilers,needlenose', 'pliers, needle nose', s)
    s = re.sub(r'\bmr.', 'mr ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub('&nbsp;', '', s)
    s = re.sub('&#39;', '', s)
    s = re.sub(r'(?<=[0-9]),[\ ]*(?=[0-9])', '', s)
    s = s.replace(";", ".")
    s = s.replace(",", ".")
    s = s.replace(":", ". ")
    s = s.replace("+", " ")
    s = re.sub(r'\bU.S.', 'US ', s)
    s = s.replace(" W x ", " ")
    s = s.replace(" H x ", " ")
    s = re.sub(' [\#]\d+[\-\d]*[\,]*', '', s)
    s = re.sub('(?<=[0-9\%])(?=[A-Z][a-z])', '. ', s)  # add dot between number and cap letter
    s = re.sub(r'(?<=\))(?=[a-zA-Z0-9])', ' ', s)  # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z0-9])(?=\()', ' ', s)  # add space between parentheses and letters
    s = re.sub('[^a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]', ' ', s)
    s = " ".join(s.split())
    s = s.replace("-"," ")
    s = toLowerCase(s)
    s = remove_stop_word(s)
    s = re.sub('\.(?=[a-z])', '. ', s)  # dots before words -> replace with spaces
    s = re.sub('(?<=[a-z][a-z][a-z])(?=[0-9])', ' ', s)
    s = re.sub('(?<=\()[a-zA-Z0-9\n\ \%\$\-\#\@\&\/\.\'\*\(\)]*(?=\))', '', s) # remover brackets
    s = s.replace("at&t", "att")
    s = s.replace("&", " and ")
    # s = s.replace("*", " x ") # replace for further removal
    s = re.sub('(?<=[a-z\ ])\/(?=[a-z\ ])', ' ', s)  # replace "/" between words with space
    s = re.sub('(?<=[a-z])\\\\(?=[a-z])', ' ', s)  # replace "/" between words with space
    s = re.sub('[^a-zA-Z0-9\ \%\$\-\@\&\/\.]', '', s)  # remove "'" and "\n" and "#" and characters
    # s = re.sub('(?<=[0-9])x(?=\ |$)', '', s)  # remove x
    # s = re.sub('(?<=[\ ])x(?=[0-9])', '', s)  # remove x
    # s = re.sub('(?<=^)x(?=[0-9])', '', s)
    s = s.replace("  ", " ")
    s = s.replace("..", ".")
    s = re.sub('\ \.', '', s)
    return s

# spelling check using Google dict for search term
def spellcheck_by_google(texts):
    if texts in google_dict.keys():
        texts = google_dict[texts]
    return texts

# mannual spell correction
def wordReplace(s):
    s = s.replace("ttt", "tt")
    s = s.replace("lll", "ll")
    s = s.replace("nnn", "nn")
    s = s.replace("rrr", "rr")
    s = s.replace("sss", "ss")
    s = s.replace("zzz", "zz")
    s = s.replace("ccc", "cc")
    s = s.replace("eee", "ee")

    s = s.replace("hinges with pishinges with pins", "hinges with pins")
    s = s.replace("virtue usa", "virtu usa")
    s = re.sub('outdoor(?=[a-rt-z])', 'outdoor ', s)
    s = re.sub(r'\bdim able\b', "dimmable", s)
    s = re.sub(r'\blink able\b', "linkable", s)
    s = re.sub(r'\bm aple\b', "maple", s)
    s = s.replace("aire acondicionado", "air conditioner")
    s = s.replace("borsh in dishwasher", "bosch dishwasher")
    s = re.sub(r'\bapt size\b', 'appartment size', s)
    s = re.sub(r'\barm[e|o]r max\b', 'armormax', s)
    s = re.sub(r' ss ', ' stainless steel ', s)
    s = re.sub(r'\bmay tag\b', 'maytag', s)
    s = re.sub(r'\bback blash\b', 'backsplash', s)
    s = re.sub(r'\bbum boo\b', 'bamboo', s)
    s = re.sub(r'(?<=[0-9] )but\b', 'btu', s)
    s = re.sub(r'\bcharbroi l\b', 'charbroil', s)
    s = re.sub(r'\bair cond[it]*\b', 'air conditioner', s)
    s = re.sub(r'\bscrew conn\b', 'screw connector', s)
    s = re.sub(r'\bblack decker\b', 'black and decker', s)
    s = re.sub(r'\bchristmas din\b', 'christmas dinosaur', s)
    s = re.sub(r'\bdoug fir\b', 'douglas fir', s)
    s = re.sub(r'\belephant ear\b', 'elephant ears', s)
    s = re.sub(r'\bt emp gauge\b', 'temperature gauge', s)
    s = re.sub(r'\bsika felx\b', 'sikaflex', s)
    s = re.sub(r'\bsquare d\b', 'squared', s)
    s = re.sub(r'\bbehring\b', 'behr', s)
    s = re.sub(r'\bcam\b', 'camera', s)
    s = re.sub(r'\bjuke box\b', 'jukebox', s)
    s = re.sub(r'\brust o leum\b', 'rust oleum', s)
    s = re.sub(r'\bx mas\b', 'christmas', s)
    s = re.sub(r'\bmeld wen\b', 'jeld wen', s)
    s = re.sub(r'\bg e\b', 'ge', s)
    s = re.sub(r'\bmirr edge\b', 'mirredge', s)
    s = re.sub(r'\bx ontrol\b', 'control', s)
    s = re.sub(r'\boutler s\b', 'outlets', s)
    s = re.sub(r'\bpeep hole', 'peephole', s)
    s = re.sub(r'\bwater pik\b', 'waterpik', s)
    s = re.sub(r'\bwaterpi k\b', 'waterpik', s)
    s = re.sub(r'\bplex[iy] glass\b', 'plexiglass', s)
    s = re.sub(r'\bsheet rock\b', 'sheetrock', s)
    s = re.sub(r'\bgen purp\b', 'general purpose', s)
    s = re.sub(r'\bquicker crete\b', 'quikrete', s)
    s = re.sub(r'\bref ridge\b', 'refrigerator', s)
    s = re.sub(r'\bshark bite\b', 'sharkbite', s)
    s = re.sub(r'\buni door\b', 'unidoor', s)
    s = re.sub(r'\bair tit\b', 'airtight', s)
    s = re.sub(r'\bde walt\b', 'dewalt', s)
    s = re.sub(r'\bwaterpi k\b', 'waterpik', s)
    s = re.sub(r'\bsaw za(ll|w)\b', 'sawzall', s)
    s = re.sub(r'\blg elec\b', 'lg', s)
    s = re.sub(r'\bhumming bird\b', 'hummingbird', s)
    s = re.sub(r'\bde ice(?=r|\b)', 'deice', s)
    s = re.sub(r'\bliquid nail\b', 'liquid nails', s)

    s = re.sub(r'\bdeck over\b', 'deckover', s)
    s = re.sub(r'\bcounter sink(?=s|\b)', 'countersink', s)
    s = re.sub(r'\bpipes line(?=s|\b)', 'pipeline', s)
    s = re.sub(r'\bbook case(?=s|\b)', 'bookcase', s)
    s = re.sub(r'\bwalkie talkie\b', '2 pair radio', s)
    s = re.sub(r'(?<=^)ks\b', 'kwikset', s)
    s = re.sub('(?<=[0-9])[\ ]*ft(?=[a-z])', 'ft ', s)
    s = re.sub('(?<=[0-9])[\ ]*mm(?=[a-z])', 'mm ', s)
    s = re.sub('(?<=[0-9])[\ ]*cm(?=[a-z])', 'cm ', s)
    s = re.sub('(?<=[0-9])[\ ]*inch(es)*(?=[a-z])', 'in ', s)

    s = re.sub(r'(?<=[1-9]) pac\b', 'pack', s)

    s = re.sub(r'\bcfl bulbs\b', 'cfl light bulbs', s)
    s = re.sub(r' cfl(?=$)', ' cfl light bulb', s)
    s = re.sub(r'candelabra cfl 4 pack', 'candelabra cfl light bulb 4 pack', s)
    s = re.sub(r'\bthhn(?=$|\ [0-9]|\ [a-rtuvx-z])', 'thhn wire', s)
    s = re.sub(r'\bplay ground\b', 'playground', s)
    s = re.sub(r'\bemt\b', 'emt electrical metallic tube', s)
    s = re.sub(r'\boutdoor dining se\b', 'outdoor dining set', s)

    external_data_dict = {'airvents': 'air vents',
                          'antivibration': 'anti vibration',
                          'autofeeder': 'auto feeder',
                          'backbrace': 'back brace',
                          'behroil': 'behr oil',
                          'behrwooden': 'behr wooden',
                          'brownswitch': 'brown switch',
                          'byefold': 'bifold',
                          'canapu': 'canopy',
                          'cleanerakline': 'cleaner alkaline',
                          'colared': 'colored',
                          'comercialcarpet': 'commercial carpet',
                          'dcon': 'd con',
                          'doorsmoocher': 'door smoocher',
                          'dreme': 'dremel',
                          'ecobulb': 'eco bulb',
                          'fantdoors': 'fan doors',
                          'gallondrywall': 'gallon drywall',
                          'geotextile': 'geo textile',
                          'hallodoor': 'hallo door',
                          'heatgasget': 'heat gasket',
                          'ilumination': 'illumination',
                          'insol': 'insulation',
                          'instock': 'in stock',
                          'joisthangers': 'joist hangers',
                          'kalkey': 'kelkay',
                          'kohlerdrop': 'kohler drop',
                          'kti': 'kit',
                          'laminet': 'laminate',
                          'mandoors': 'main doors',
                          'mountspacesaver': 'mount space saver',
                          'reffridge': 'refrigerator',
                          'refrig': 'refrigerator',
                          'reliabilt': 'reliability',
                          'replaclacemt': 'replacement',
                          'searchgalvanized': 'search galvanized',
                          'seedeater': 'seed eater',
                          'showerstorage': 'shower storage',
                          'straitline': 'straight line',
                          'subpumps': 'sub pumps',
                          'thromastate': 'thermostat',
                          'topsealer': 'top sealer',
                          'underlay': 'underlayment',
                          'vdk': 'bdk',
                          'wallprimer': 'wall primer',
                          'weedbgon': 'weed b gon',
                          'weedeaters': 'weed eaters',
                          'weedwacker': 'weed wacker',
                          'wesleyspruce': 'wesley spruce',
                          'worklite': 'work light'}

    for word in external_data_dict.keys():
        s = re.sub(r'\b' + word + r'\b', external_data_dict[word], s)
    return s

# Lammertizer
def lemmatizer(texts):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokenizer.tokenize(texts)]
    return " ".join(tokens)

# Stemmer
def stemming(texts, type):
    stemmer = None
    if type == 'snowball':
        stemmer = nltk.stem.SnowballStemmer("english")
    elif type == 'porter':
        stemmer = nltk.stem.PorterStemmer()
    tokens = [stemmer.stem(token) for token in texts.split(" ")]
    return " ".join(tokens)

# for title
def title_processing(texts):
    texts = removeSpecialCharacters(texts)
    texts = textToDigits(texts)
    texts = removeDigitCommas(texts)
    return texts

# for description
def description_processing(texts):
    texts = removeSpecialCharacters(texts)
    texts = textToDigits(texts)
    texts = removeDigitCommas(texts)
    texts = parseHtml(texts)
    return texts

# for serach term
def serach_term_processing(texts):
    texts = toLowerCase(texts)
    texts = textToDigits(texts)
    texts = removeDigitCommas(texts)
    texts = wordReplace(texts)
    texts = spellcheck_by_google(texts)
    return texts

def attr_bullets_processing(df_attr):
    df_attr['product_uid']=df_attr['product_uid'].fillna(0)
    df_attr['value']=df_attr['value'].fillna("")
    df_attr['name']=df_attr['name'].fillna("")
    dict_attr={}
    for product_uid in list(set(list(df_attr['product_uid']))):
        dict_attr[int(product_uid)]={'product_uid':int(product_uid),'attribute_bullets':[]}

    for i in range(0,len(df_attr['product_uid'])):
#         if (i % 100000)==0:
#             print("Read",i,"out of", len(df_attr['product_uid']), "rows in attributes.csv in", round((time()-t0)/60,1) ,'minutes')
        if df_attr['name'][i][0:6]=="Bullet":
            dict_attr[int(df_attr['product_uid'][i])]['attribute_bullets'].append(df_attr['value'][i])

    if 0 in dict_attr.keys():
        del(dict_attr[0])
                        
    for item in dict_attr.keys():
        if len(dict_attr[item]['attribute_bullets'])>0:
            dict_attr[item]['attribute_bullets']=". ".join(dict_attr[item]['attribute_bullets'])
            dict_attr[item]['attribute_bullets']+="."
        else:
            dict_attr[item]['attribute_bullets']=""
    df_attr_bullets=pd.DataFrame(dict_attr).transpose()
    df_attr_bullets['attribute_bullets']=df_attr_bullets['attribute_bullets'].map(lambda x: x.replace("..",".").encode('utf-8'))
    df_attr_bullets['attribute_bullets'] = df_attr_bullets['attribute_bullets'].str.decode("utf-8")

    return df_attr_bullets


# for stem and lemminaztion
def word_processing(s):
    s = lemmatizer(s)
    s = stemming(s, "snowball")
    return s

replace_material_dict={'aluminium': 'aluminum',
    'medium density fiberboard': 'mdf',
    'high density fiberboard': 'hdf',
    'fiber reinforced polymer': 'frp',
    'cross linked polyethylene': 'pex',
    'poly vinyl chloride': 'pvc',
    'thermoplastic rubber': 'tpr',
    'poly lactic acid': 'pla',
    'acrylonitrile butadiene styrene': 'abs',
    'chlorinated poly vinyl chloride': 'cpvc',
    'Medium Density Fiberboard (MDF)': 'mdf',
    'High Density Fiberboard (HDF)': 'hdf',
    'Fibre Reinforced Polymer (FRP)': 'frp',
    'Acrylonitrile Butadiene Styrene (ABS)': 'abs',
    'Cross-Linked Polyethylene (PEX)': 'pex',
    'Chlorinated Poly Vinyl Chloride (CPVC)': 'cpvc',
    'PVC (vinyl)': 'pvc',
    'Thermoplastic rubber (TPR)': 'tpr',
    'Poly Lactic Acid (PLA)': 'pla',
    '100% Polyester': 'polyester',
    '100% UV Olefin': 'olefin',
    '100% BCF Polypropylene': 'polypropylene',
    '100% PVC': 'pvc'
    }

# for extracting material
def parse_material(s):
    s = s.replace("Other", "")
    s = s.replace("*", "")
    s = re.sub('&amp;', '&', s)
    s = re.sub('&nbsp;', '', s)
    s = re.sub('&#39;', '', s)
    s = s.replace("-", " ")
    s = s.replace("+", " ")
    s = re.sub(r'(?<=[a-zA-Z])\/(?=[a-zA-Z])', ' ', s)
    s = re.sub(r'(?<=\))(?=[a-zA-Z0-9])', ' ', s)  # add space between parentheses and letters like
    s = re.sub(r'(?<=[a-zA-Z0-9])(?=\()', ' ', s)  # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z][\.\,])(?=[a-zA-Z])', ' ', s)  # add space after dot or colon between letters
    s = re.sub('[^a-zA-Z0-9\n\ ]', '', s)
    if s in replace_material_dict.keys():
        s =replace_material_dict[s]
    return s.lower()

del_brand_list = ['aaa','off','impact','square','shelves','finish','ring','flood','dual','ball','cutter',\
'max','off','mat','allure','diamond','drive', 'edge','anchor','walls','universal','cat', 'dawn','ion','daylight',\
'roman', 'weed eater', 'restore', 'design', 'caddy', 'pole caddy', 'jet', 'classic', 'element', 'aqua',\
'terra', 'decora', 'ez', 'briggs', 'wedge', 'sunbrella',  'adorne', 'santa', 'bella', 'duck', 'hotpoint',\
'duck', 'tech', 'titan', 'powerwasher', 'cooper lighting', 'heritage', 'imperial', 'monster', 'peak',
'bell', 'drive', 'trademark', 'toto', 'champion', 'shop vac', 'lava', 'jet', 'flood', \
'roman', 'duck', 'magic', 'allen', 'bunn', 'element', 'international', 'larson', 'tiki', 'titan', \
 'space saver', 'cutter', 'scotch', 'adorne', 'ball', 'sunbeam', 'fatmax', 'poulan', 'ring', 'sparkle', 'bissell', \
 'universal', 'paw', 'wedge', 'restore', 'daylight', 'edge', 'americana', 'wacker', 'cat', 'allure', 'bonnie plants', \
 'troy', 'impact', 'buffalo', 'adams', 'jasco', 'rapid dry', 'aaa', 'pole caddy', 'pac', 'seymour', 'mobil', \
 'mastercool', 'coca cola', 'timberline', 'classic', 'caddy', 'sentry', 'terrain', 'nautilus', 'precision', \
 'artisan', 'mural', 'game', 'royal', 'use', 'dawn', 'task', 'american line', 'sawtrax', 'solo', 'elements', \
 'summit', 'anchor', 'off', 'spruce', 'medina', 'shoulder dolly', 'brentwood', 'alex', 'wilkins', 'natural magic', \
 'kodiak', 'metro', 'shelter', 'centipede', 'imperial', 'cooper lighting', 'exide', 'bella', 'ez', 'decora', \
 'terra', 'design', 'diamond', 'mat', 'finish', 'tilex', 'rhino', 'crock pot', 'legend', 'leatherman', 'remove', \
 'architect series', 'greased lightning', 'castle', 'spirit', 'corian', 'peak', 'monster', 'heritage', 'powerwasher',\
 'reese', 'tech', 'santa', 'briggs', 'aqua', 'weed eater', 'ion', 'walls', 'max', 'dual', 'shelves', 'square',\
 'hickory', "vikrell", "e3", "pro series", "keeper", "coastal shower doors", 'cadet','church','gerber','glidden',\
 'cooper wiring devices', 'border blocks', 'commercial electric', 'pri','exteria','extreme', 'veranda',\
 'gorilla glue','gorilla','shark','wen']

replace_brand_dict={
'acurio latticeworks': 'acurio',
'american kennel club':'akc',
'amerimax home products': 'amerimax',
'barclay products':'barclay',
'behr marquee': 'behr',
'behr premium': 'behr',
'behr premium deckover': 'behr',
'behr premium plus': 'behr',
'behr premium plus ultra': 'behr',
'behr premium textured deckover': 'behr',
'behr pro': 'behr',
'bel air lighting': 'bel air',
'bootz industries':'bootz',
'campbell hausfeld':'campbell',
'columbia forest products': 'columbia',
'essick air products':'essick air',
'evergreen enterprises':'evergreen',
'feather river doors': 'feather river',
'gardner bender':'gardner',
'ge parts':'ge',
'ge reveal':'ge',
'gibraltar building products':'gibraltar',
'gibraltar mailboxes':'gibraltar',
'glacier bay':'glacier',
'great outdoors by minka lavery': 'great outdoors',
'hamilton beach': 'hamilton',
'hampton bay':'hampton',
'hampton bay quickship':'hampton',
'handy home products':'handy home',
'hickory hardware': 'hickory',
'home accents holiday': 'home accents',
'home decorators collection': 'home decorators',
'homewerks worldwide':'homewerks',
'klein tools': 'klein',
'lakewood cabinets':'lakewood',
'leatherman tool group':'leatherman',
'legrand adorne':'legrand',
'legrand wiremold':'legrand',
'lg hausys hi macs':'lg',
'lg hausys viatera':'lg',
'liberty foundry':'liberty',
'liberty garden':'liberty',
'lithonia lighting':'lithonia',
'loloi rugs':'loloi',
'maasdam powr lift':'maasdam',
'maasdam powr pull':'maasdam',
'martha stewart living': 'martha stewart',
'merola tile': 'merola',
'miracle gro':'miracle',
'miracle sealants':'miracle',
'mohawk home': 'mohawk',
'mtd genuine factory parts':'mtd',
'mueller streamline': 'mueller',
'newport coastal': 'newport',
'nourison overstock':'nourison',
'nourison rug boutique':'nourison',
'owens corning': 'owens',
'premier copper products':'premier',
'price pfister':'pfister',
'pride garden products':'pride garden',
'prime line products':'prime line',
'redi base':'redi',
'redi drain':'redi',
'redi flash':'redi',
'redi ledge':'redi',
'redi neo':'redi',
'redi niche':'redi',
'redi shade':'redi',
'redi trench':'redi',
'reese towpower':'reese',
'rheem performance': 'rheem',
'rheem ecosense': 'rheem',
'rheem performance plus': 'rheem',
'rheem protech': 'rheem',
'richelieu hardware':'richelieu',
'rubbermaid commercial products': 'rubbermaid',
'rust oleum american accents': 'rust oleum',
'rust oleum automotive': 'rust oleum',
'rust oleum concrete stain': 'rust oleum',
'rust oleum epoxyshield': 'rust oleum',
'rust oleum flexidip': 'rust oleum',
'rust oleum marine': 'rust oleum',
'rust oleum neverwet': 'rust oleum',
'rust oleum parks': 'rust oleum',
'rust oleum professional': 'rust oleum',
'rust oleum restore': 'rust oleum',
'rust oleum rocksolid': 'rust oleum',
'rust oleum specialty': 'rust oleum',
'rust oleum stops rust': 'rust oleum',
'rust oleum transformations': 'rust oleum',
'rust oleum universal': 'rust oleum',
'rust oleum painter touch 2': 'rust oleum',
'rust oleum industrial choice':'rust oleum',
'rust oleum okon':'rust oleum',
'rust oleum painter touch':'rust oleum',
'rust oleum painter touch 2':'rust oleum',
'rust oleum porch and floor':'rust oleum',
'salsbury industries':'salsbury',
'simpson strong tie': 'simpson',
'speedi boot': 'speedi',
'speedi collar': 'speedi',
'speedi grille': 'speedi',
'speedi products': 'speedi',
'speedi vent': 'speedi',
'pass and seymour': 'seymour',
'pavestone rumblestone': 'rumblestone',
'philips advance':'philips',
'philips fastener':'philips',
'philips ii plus':'philips',
'philips manufacturing company':'philips',
'safety first':'safety 1st',
'sea gull lighting': 'sea gull',
'scott':'scotts',
'scotts earthgro':'scotts',
'south shore furniture': 'south shore',
'tafco windows': 'tafco',
'trafficmaster allure': 'trafficmaster',
'trafficmaster allure plus': 'trafficmaster',
'trafficmaster allure ultra': 'trafficmaster',
'trafficmaster ceramica': 'trafficmaster',
'trafficmaster interlock': 'trafficmaster',
'thomas lighting': 'thomas',
'unique home designs':'unique home',
'veranda hp':'veranda',
'whitehaus collection':'whitehaus',
'woodgrain distritubtion':'woodgrain',
'woodgrain millwork': 'woodgrain',
'woodford manufacturing company': 'woodford',
'wyndham collection':'wyndham',
'yardgard select': 'yardgard',
'yosemite home decor': 'yosemite'
} # replace with shorter words


# for extracting brands
def parse_brand(s):
    s = s.replace(".N/A", "")
    s = s.replace("N.A.", "")
    s = s.replace("n/a", "")
    s = s.replace("Generic Unbranded", "")
    s = s.replace("Unbranded", "")
    s = s.replace("Generic", "")
    s = s.lower()
    s = re.sub('&amp;', '&', s)
    s = re.sub('&nbsp;', '', s)
    s = re.sub('&#39;', '', s)
    s = s.replace("-", " ")
    s = s.replace("+", " ")
    s = re.sub(r'(?<=[a-zA-Z])\/(?=[a-zA-Z])', ' ', s)
    s = re.sub(r'(?<=\))(?=[a-zA-Z0-9])', ' ', s)  # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z0-9])(?=\()', ' ', s)  # add space between parentheses and letters
    s = re.sub(r'(?<=[a-zA-Z][\.\,])(?=[a-zA-Z])', ' ', s)  # add space after dot or colon between letters
    s = re.sub('[^a-zA-Z0-9\n\ ]', '', s)
    if s in replace_brand_dict.keys():
        return replace_brand_dict[s]
    # if s in del_brand_list:
    #     s = ""
    return s

'''
Functions for feature Extraction
'''

def query_in_text(str1, str2):
    """
    Returns 1 if str1 (query) is found in str2, 0 otherwise.
    """
    output=0
    if len(str1.split())>0:
        if str1 in str2:
             if re.search(r'\b'+str1+r'\b',str2)!=None:
                    output=1
    return output
        
def str_common_word(str1, str2, minLength=1, string_only=False):
    """
The function that return a bundle of count features for the pair of strings:
- number of unique words in intersection
- number of total words in intersection
- number of letters in unique words in intersection
- ratio of common words to all words in str1 (query)
- ratio of the number of letters in common words to the total number of letters in str1 (query)
Also, the common words are returned as a string.
Example:
str1 = "table with cover"
str2 = "wood cover"
the function returns (1, 1, 5, 0.3333333333333333, 0.35714285714285715, 'cover')
   """ 
    word_list=[]
    num=0
    total_entries=0
    cnt_letters=0
    cnt_unique_letters=0
    all_num=0
    all_total_entries=0
    all_cnt_letters=0
    for word in str1.split():
         if len(word)>=minLength:
                if string_only==False or len(re.findall(r'\d+', word))==0:
                    if (' '+word+' ') in (' '+str2+' '):
                        num+=1
                        total_entries+=(' '+str2+' ').count(' '+word+' ')
                        cnt_letters+=(' '+str2+' ').count(' '+word+' ') * (len(word))
                        cnt_unique_letters+=(len(word))
                        word_list.append(word)
                all_num+=1
                all_total_entries+=1
                all_cnt_letters+=len(word)
    
    if all_num==0:
        ratio_num=0
    else:
        ratio_num=1.0*num/all_num
    
    if all_cnt_letters==0:
        ratio_letters=0
    else:
        ratio_letters=1.0*cnt_unique_letters/all_cnt_letters              
    return num, total_entries, cnt_unique_letters, ratio_num, ratio_letters, " ".join(word_list)


def str_common_digits(str1, str2):
    """
    Similar to str_common_words(), but designed specifically for digits.
    """
    found=0
    found_words_only=0
    digits_in_query=list(set(re.findall(r'\d+\/\d+|\d+\.\d+|\d+', str1)))
    digits_in_text=re.findall(r'\d+\/\d+|\d+\.\d+|\d+', str2)
    len1=len(digits_in_query)
    len2=len(digits_in_text)
    for digit in digits_in_query:
        if digit in digits_in_text:
            found+=1             
    if len1==0:
        ratio=0.
    else:
        ratio=found/len1       
    if (len1 + len2)==0:
        jaccard=0.
    else:
        jaccard=1.0*found/(len1 + len2)
    return len1, len2, found, ratio, jaccard


def seq_matcher(s1,s2):
    """
Return ratio and scaled ratio from difflib.SequenceMatcher()
    """
    seq=difflib.SequenceMatcher(None, s1,s2)
    rt=round(seq.ratio(),7)
    l1=len(s1)
    l2=len(s2)
    if len(s1)==0 or len(s2)==0:
        rt=0
        rt_scaled=0
    else:
        rt_scaled=round(rt*max(l1,l2)/min(l1,l2),7)
    return rt, rt_scaled

def calculate_n_sim(df_all,model_list):
    st = df_all["search_term_stemmed"]
    pt = df_all["product_title_stemmed"]
    pd = df_all["product_description_stemmed"]
    br = df_all["product_brand"]
    mr = df_all["materials"]
    ab = df_all["attribute_bullets_stemmed"]
    
    n_sim=list()
    count = 0
    for model in model_list:
        print('#####################')
        print('model vocab')
#         print(model.wv.vocab)
        n_sim_pt=list()
        for i in range(len(st)):
            w1=st[i].split()
            w2=pt[i].split()
            d1=[]
            d2=[]
            for j in range(len(w1)):
                if w1[j] in model.wv.vocab:
                    d1.append(w1[j])
            for j in range(len(w2)):
                if w2[j] in model.wv.vocab:
                    d2.append(w2[j])
            if d1==[] or d2==[]:
                n_sim_pt.append(0)
            else:    
                n_sim_pt.append(model.wv.n_similarity(d1,d2))
        n_sim.append(n_sim_pt)
        print()
        print('n_sim ', count,'search_term & title')
        count += 1
        
        n_sim_pd=list()
        for i in range(len(st)):
            w1=st[i].split()
            w2=pd[i].split()
            d1=[]
            d2=[]
            for j in range(len(w1)):
                if w1[j] in model.wv.vocab:
                    d1.append(w1[j])
            for j in range(len(w2)):
                if w2[j] in model.wv.vocab:
                    d2.append(w2[j])
            if d1==[] or d2==[]:
                n_sim_pd.append(0)
            else:    
                n_sim_pd.append(model.wv.n_similarity(d1,d2))
        n_sim.append(n_sim_pd)
        print()
        print('n_sim ', count,'search_term & description')
        count += 1
        
        n_sim_pd=list()
        for i in range(len(st)):
            w1=st[i].split()
            w2=ab[i].split()
            d1=[]
            d2=[]
            for j in range(len(w1)):
                if w1[j] in model.wv.vocab:
                    d1.append(w1[j])
            for j in range(len(w2)):
                if w2[j] in model.wv.vocab:
                    d2.append(w2[j])
            if d1==[] or d2==[]:
                n_sim_pd.append(0)
            else:    
                n_sim_pd.append(model.wv.n_similarity(d1,d2))
        n_sim.append(n_sim_pd)
        print()
        print('n_sim ', count,'search_term & attr bullet')
        count += 1
        
        n_sim_all=list()
        for i in range(len(st)):
            w1=st[i].split()
            w2=pt[i].split()+pd[i].split()+br[i].split()+mr[i].split()+ab[i].split()
            d1=[]
            d2=[]
            for j in range(len(w1)):
                if w1[j] in model.wv.vocab:
                    d1.append(w1[j])
            for j in range(len(w2)):
                if w2[j] in model.wv.vocab:
                    d2.append(w2[j])
            if d1==[] or d2==[]:
                n_sim_all.append(0)
            else:    
                n_sim_all.append(model.wv.n_similarity(d1,d2))
        n_sim.append(n_sim_all)
        print()
        print('n_sim ', count,'search_term & all other texts')
        count += 1

    
        n_sim_ptpd=list()
        for i in range(len(st)):
            w1=pt[i].split()
            w2=pd[i].split()
            d1=[]
            d2=[]
            for j in range(len(w1)):
                if w1[j] in model.wv.vocab:
                    d1.append(w1[j])
            for j in range(len(w2)):
                if w2[j] in model.wv.vocab:
                    d2.append(w2[j])
            if d1==[] or d2==[]:
                n_sim_ptpd.append(0)
            else:    
                n_sim_ptpd.append(model.wv.n_similarity(d1,d2))
        n_sim.append(n_sim_ptpd)
        print()
        print('n_sim ', count,'description & title')
        count += 1
    return n_sim
    
def get_word2vec_n_sim(df_all):
    #build a set of sentenxes
    st = df_all["search_term_stemmed"]
    pt = df_all["product_title_stemmed"]
    pd = df_all["product_description_stemmed"]
    br = df_all["product_brand"]
    mr = df_all["materials"]
    ab = df_all["attribute_bullets_stemmed"]
    
#st + pt +pd + ab vocab
# example of t 
# [['l', 'bracket'], ['deckov'], ['rain', 'shower', 'head'], ['shower', 'onli', 'faucet'], 
# ['convect', 'otr'], ['microwav', 'over', 'stove'], ['microwav']...]
    t = list()
    for i in range(len(st)):
        p = st[i].split()
        t.append(p)

    for i in range(len(pt)):
        p = pt[i].split()
        t.append(p)
    
    for i in range(len(pd)):
        p = pd[i].split()
        t.append(p)
     
    
    for i in range(len(ab)):
        p = ab[i].split()
        t.append(p)
    print("first vocab")
#     print(len(t))
#     print('t',t)
#st conc pt conc pd vocab
# example of t1:
# [['l', 'bracket', 'simpson', 'strong', 'tie', '12', 'gaug', 'angl', 'angl', 'make', 'joint', 'stronger', 'also',
# 'provid', 'consist', 'straight', 'corner', 'simpson', 'strong', 'tie', 'offer', 'wide', 'varieti', 'angl', 
# 'various', 'size', 'thick', 'handl', 'light', 'duti', 'job', 'project', 'structur', 'connect', 'need', 'bent',...],
    t1 = list()
    for i in range(len(st)):
        p = st[i].split()+pt[i].split()+pd[i].split()+br[i].split()+mr[i].split()+ab[i].split()
        t1.append(p)
    print("second vocab")
#     print(len(t1))
#     print('t1',t1[:1])
    
    model0 = gensim.models.Word2Vec(t, sg=1, window=10, sample=1e-5, negative=5, size=300)
    model1 = gensim.models.Word2Vec(t1, sg=1, window=10, sample=1e-5, negative=5, size=300)
    
    #for each model calculate features^ n_similarity between st and something else
    model_list=[model0,model1]
    n_sim = calculate_n_sim(df_all,model_list)
    print("model features done")
    return n_sim



def to_tfidf(text,dictionary_tfidf,corpus):
    tfidf = TfidfModel(corpus)
    res = tfidf[dictionary_tfidf.doc2bow(list(tokenize(text, errors='ignore')))]
    return res

def calculate_tfidf_cos_sim(text1,text2,dictionary_tfidf,corpus):

    tfidf1 = to_tfidf(text1,dictionary_tfidf,corpus)
    tfidf2 = to_tfidf(text2,dictionary_tfidf,corpus)
    index = MatrixSimilarity([tfidf1],num_features=len(dictionary_tfidf))
    sim = index[tfidf2]
    return float(sim[0])

def calculable_features(df):
    result = []
    for col in df.columns:    
        if df[col].dtypes != object:
            result.append(col)
    return result

def cross_validation_with_selected_features(features, model, df_all):
    # generate train/test with selected features
    test_ids, y_train, x_train, x_test = generate_train_test_data(features, df_all)    
    # cross validation
    cv = ShuffleSplit(test_size=0.3, random_state=0)
    test_score = np.sqrt(-cross_val_score(model, x_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
    return test_score

def rank_features(features, model, df_all):
    '''
    features: all numeric features
    model: model
    '''
    result = []
    test_ids, y_train, x_train, x_test = generate_train_test_data(features, df_all)
    feature_importance = list(model.fit(x_train, y_train).feature_importances_)
    feature_feature_importance = pd.DataFrame({'feature': features[2:], 'feature_importance':feature_importance})
    features = feature_feature_importance.sort_values('feature_importance', ascending=False)
    
    return features

# cr.liuchang
def generate_train_test_data(features, df):
    '''
    Generate train/test data based on selected features.
    '''
    df_sub = df[features]
    # Seperate train and test data
    df_test = df_sub.loc[pd.isnull(df_sub['relevance'])]
    df_train = df_sub.loc[pd.notnull(df_sub['relevance'])]
    # Record test ids
    test_ids = df_test['id']
    # Get training label: relevance
    y_train = df_train['relevance'].values
    # Delete id and label
    X_train = df_train.drop(['id', 'relevance'], axis=1).values
    X_test = df_test.drop(['id', 'relevance'], axis=1).values
    
    return test_ids, y_train, X_train, X_test

def num_of_features_cv(model, feature_len, input_features, df_all):
    '''
    model: base learners
    feature_len: 
    '''
    test_scores = []
    num_of_features = list(range(2, feature_len + 1))
    for n in range(2, feature_len + 1):
        features = input_features[:n]
        features.extend(['id', 'relevance'])    
        test_score = cross_validation_with_selected_features(features, model, df_all)
        test_scores.append(np.mean(test_score))
    return num_of_features, test_scores

def generate_feature_importances_dataframes(base_learners, numeric_features, df_all):
    features_importances = pd.DataFrame()
    for model_name, model in base_learners:
        dataframe = rank_features(numeric_features, model, df_all)
        # reset index
        dataframe = dataframe.reset_index(drop=True)
        col_feature = 'feature_' + model_name
        col_feature_importance = 'importances_' + model_name
        features_importances[col_feature] = dataframe['feature']
        features_importances[col_feature_importance] = dataframe['feature_importance']        
    return features_importances
    
def gridSearch_tuning_estimators(estimators, params, X_train, y_train):
    for model_name, model in estimators:
        tuned = GridSearchCV(estimator=model, param_grid=params,cv=5)
        tuned.fit(X_train, y_train) # no attribute 'feature_importances_'
        model.set_params(**tuned.best_params_)
