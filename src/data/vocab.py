from typing import List, Tuple

class Vocab:

    def build_token_to_id_vocab(self,sentences: List[str], specials:Tuple[str,...]=('<pad>', '<bos>', '<eos>', '<unk>')) -> dict[str,int]:
    
        self.token_to_ids = {}

        for i,special in enumerate(specials):
            self.token_to_ids[special] = i
        
        length = len(self.token_to_ids)

        for sentence in sentences:
            words = sentence.split(' ')
            for word in words:
                if word not in self.token_to_ids:
                    self.token_to_ids[word] = length
                    length = length + 1

        return self.token_to_ids
       




    

