from typing import List, Tuple

class Vocab:

    def __init__(self):
        self.token_to_id: dict[str,int] = {}
        self.id_to_token: dict[int,str] = {}

    def build_token_to_id_vocab(self,sentences: List[str], specials:Tuple[str,...]=('<pad>', '<bos>', '<eos>', '<unk>')) -> dict[str,int]:
    
        self.token_to_id = {}

        for i,special in enumerate(specials):
            self.token_to_id[special] = i
        
        length = len(self.token_to_id)


        for sentence in sentences:
            words = sentence.split(' ')
            for word in words:
                if word not in self.token_to_id:
                    self.token_to_id[word] = length
                    length = length + 1

        return self.token_to_id
       

    
    def build_id_to_token_vocab(self,token_to_id: dict[str,int]) -> dict[int,str]:

        self.id_to_token = {value:key for key,value in token_to_id.items()}
        return self.id_to_token


class Encode:

    def __init__(self, token_to_id:dict[str,int]):
        self.token_to_id = token_to_id

    def sentence_to_ids(self,sentence:str)->List[int]:

        ids = []
        tokens = sentence.split(' ')
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            elif(token):
                ids.append(self.token_to_id['<unk>'])
        return ids
    
class Decode:

    def __init__(self, id_to_token: dict[int,str]):
        self.id_to_token = id_to_token

    def ids_to_tokens(self,ids:List[int]) -> List[str]:

        tokens = []
        for idx in ids:
            tokens.append(self.id_to_token[idx])
        return tokens


    
def pad_id_sequence(ids:List[int], max_len:int, pad_id:int) -> List[int]:
    
    ids_length = len(ids)
    if max_len < ids_length:
        ids = ids[:max_len]
    elif max_len > ids_length:
        for i in range((max_len - ids_length)):
            ids.append(pad_id)
    else:
        return ids
    
    return ids
