import nltk
import re
import pprint
from nltk import Tree

new_patterns = """
                NP:    
                {<JJ|VPN>*<NN|NNS|NNP|NNPS>+<IN><PRP>*<JJ>*<NN|NNS|NNP|NNPS>}
                {<JJ>+<CC>*<JJ|VPN>+<NN|NNS|NNP|NNPS>+}
                {<JJ|VPN>*<NN|NNS|NNP|NNPS>+}
                {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
                {<DT><WP><VBP>*<RB>*<VBN><IN><NN|NNS|NNP|NNPS>}
                """
clothing_map_coarse={
    'blouse':'blouse',
    'vest':'top',
    'shoes':'footwear',
    'shirt':'top',
    'hoodies':'hoodie',
    'hoodie':'hoodie',
    'top':'top',
    'tshirt':'top',
    'sweater':'hoodie',
    'coat':'jacket',
    'jeans':'pants',
    'pants':'pants',
    'short':'shorts',
    'shorts':'shorts',
    'dress':'dress',
    'socks':'footwear',
    'boots':'footwear',
    'sweatshirt':'hoodie',
    'suit':'jacket',
    'jacket':'jacket',
    'skirt':'dress',
    'sandals':'footwear',
    'sneakers':'footwear',
    'jean':'pants',
    'leggings':'pants',
    'tee':'top',
    'polo':'top',}

color_map_coarse={
    'maroon': 'brown',
    'red':'red',
    'pink':'pink',
    'orange':'orange',
    'purple':'purple',
    'yellow':'yellow',
    'tan':'brown',
    'khaki':'brown',
    'brown':'brown',
    'gold':'yellow',
    'silver':'gray',
    'blonde':'yellow',
    'dark':'dark',
    'blue':'blue',
    'dark-blue':'blue',
    'white':'white',
    'black':'black',
    'colored':'color',
    'gray':'gray',
    'grey':'gray',
    'color':'color',
    'green':'green',
    'tank':'tank',
    'light': 'light', 
}
attr_map_coarse = {
    'sleeves':'sleeve',
    'sleeved':'sleeve',
    
    'pocket':'pocket',
    'pockets':'pocket',
    'asian':'asian',
    'design':'design',
    'open':'open',
    'hooded':'hood',
    'hood':'hood',
    'loose':'loose',
    'straps':'strap',
    'pattern':'pattern',
    'skinny':'skinny',
    'thin':'skinny',
    'tennis':'tennis',
    'patterned':'pattern',
    'floral':'floral',
    'shortsleeved':'shortsleeve',
    'leather':'leather',
    'print':'print',
    'straight':'straight',
    'stripe':'stripe',
    'strap':'strap',
    'tight':'tight',
    'striped':'stripe',
    'sleeve':'sleeve',
    'sleeved':'sleeve',
    'sleeves':'sleeve',
    'young':'young',
    'large':'large',
    'small':'small',
    'sleeveless':'sleveeless',
    'long':'long',
    'button':'button',
    'stripes':'stripe',
    'collar':'collar',
    'collared':'collar',
    'striped':'stripe',
    'denim':'denim',
    'plaid':'plaid',
} 
acc_map_coarse = {
    'ponytail':'ponytail',
    'highheel':'highheel',
    'heel':'highheel',
    'pack':'backpack',
    'hat':'hat',
    'purse':'purse',
    'watch':'watch',
    'glasses':'glasses',
    'cap':'hat',
    'book':'book',
    'bags':'bag',
    'bag':'bag',
    'handbag':'bag',
    'backpack':'backpack',
    'phone':'phone',
    'scarf':'scarf',
    'scarfs':'scarf',
    'belt':'belt',
    'tie':'tie',
    'camera':'camera',
    'umbrella':'umbrella',
    'cell':'phone',
    'sunglasses':'sunglasses',
}

class NPExtractor:
    def __init__(self):
        self.NPChunker = nltk.RegexpParser(new_patterns)
        self.JJ_list = set(list(color_map_coarse.keys()) + list(attr_map_coarse.keys()))
    

    def prepare_text(self, input):
        sentences = nltk.sent_tokenize(input.lower())
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [self.pos_tag(sent) for sent in sentences]
        sentences = [self.NPChunker.parse(sent) for sent in sentences]
        return sentences

    def pos_tag(self, sent):
        out = nltk.pos_tag(sent)
        ret = []
        for i, (word,label) in enumerate(out):
            if word in self.JJ_list:
                out[i] = (word, 'JJ')
            if label.startswith("PRP"):
                out[i] = (word, "PRP")
            if word == "pair":
                out[i] = (word, "UNK")
        return out
            
            
    
    
    def parsed_text_to_NP(self, sentences):
        nps = []
        for sent in sentences:
            tree = self.NPChunker.parse(sent)
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    t = subtree
                    t = ' '.join(word for word, tag in t.leaves())
                    nps.append(t)
        return nps


    def sent_parse(self,input):
        sentences = self.prepare_text(input)
        nps = self.parsed_text_to_NP(sentences)
        if len(nps) == 0:
            return ['<UNK>']
        return nps