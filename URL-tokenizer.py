"""
 This is an implementation of URL-Tokenizer algorithm published 
 @ WI-IAT '21: IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology
 @ by author Eint Sandi Aung
 @ The paper is available on https://doi.org/10.1145/3486622.3493983
"""
import enchant
from wordsegment import load, segment
from transformers import BertTokenizer
import re
from urllib.parse import unquote_plus,urlparse
def ws_bt_ec(urls,mode):
    tokenizer_ = BertTokenizer.from_pretrained("bert-base-uncased")
    tok_url=[]
    d = enchant.Dict("en_US")   # Using 'en_US' dictionary
    load() # Load WordSegment
    for url in urls:
        url=unquote_plus(url, encoding='utf-8')
        parsedURL=urlparse(url)
        if mode=='url': # For URL
            url=parsedURL.scheme.lower()+'://'+parsedURL.netloc.lower()+parsedURL.path.lower()
        elif mode=='domain':    # For Domain
            url=parsedURL.scheme.lower()+'://'+parsedURL.netloc.lower()

        tmp_words=re.split(r"[-_;:,.=?@\s$?+!*\'()[\]{}|\"%~#<>/]",url)   #Normal Tokenization by Special Characters
        sp_chars=[] # For Speical Characters Extraction
        tok_words=[]    # For Extracted Words
        w_list=[]   # For Validated Word 
        res_w_list=[]   # For Final Word after recursion
    
        # Truncated Words
        for i in url:   
            if not i.isalnum(): # Extracting Speical Characters
                sp_chars.append(i)
        for each in tmp_words:  # Words w/o Special Characters
            if len(each)>0 and len(each)<=50: # Set recursive length
                tok_words.append(each.lower())
            elif len(each)>50: # Set recursive length
                tok_words.append(each[:50].lower())
            if len(sp_chars) >0:    # Add Special Characters back
                tok_words.append(str(sp_chars[0]))
                sp_chars.remove(sp_chars[0])
        
        # Recursive Tokenization
        for word in tok_words:
            if  str(word).encode().isalnum(): # Check alphanumeric character a-zA-Z0-9
                words=segment(word) # Word Segmentation by WordSegment
                for each in words:  
                    # Validating English word or not
                    if d.check(each) or d.check(str(each).upper()):
                        rec_ret=[each]
                    else:
                        # Recursive Tokenization of substring
                        rec_ret=re.split(r"[-]",recursive_tz(each,tokenizer_)) 

                    # Prefixing word which is neither the first word nor an English word, into ##word, similar to Bert tokenizer 
                    # to indicate the contiuation of words
                    for i in rec_ret:   
                        # English and First word
                        if (d.check(i) or d.check(str(i).upper())) and (i ==rec_ret[0]):
                            w_list.append(str(i).lower())
                        # English but NOT First word
                        elif (d.check(i) or d.check(str(i).upper())) and (i !=rec_ret[0]):
                            w_list.append(str(i).lower())
                        # NOT English but First word
                        elif (not (d.check(i) or d.check(str(i).upper()))) and (i ==rec_ret[0]):
                            w_list.append(str(i).lower())
                        # Neither English nor First word
                        else:
                            w_list.append('##'+str(i).lower())
            else: #check non-alphanumeric character e.g., ://.
                w_list.append(word)
        
        # Optional
        isReplaceUnk=False
        if isReplaceUnk==True:
            # Replacing "Neither Special Characters nor Alphanumeric" words into [UNK]
            string_check= re.compile('[-_;:,.=@\s$?+!*\'()[\]{}|\"%~#<>/]')  # Special Characters
            for each in w_list:
                # Neither Special Characters nor Alphanumeric
                if (string_check.search(each)==None) and (not str(each).encode().isalnum()):
                    res_w_list.append('[UNK]')
                else:
                    res_w_list.append(each)
            tok_url.append(res_w_list)
            return tok_url

    return w_list

def recursive_tz(words:str,tokenizer_):
    word_tz=tokenizer_.tokenize(words)
    if len(word_tz)==1:
        return word_tz[0]
    else:
        rec_str=words[len(word_tz[0]):]
        return word_tz[0]+'-'+recursive_tz(rec_str,tokenizer_)