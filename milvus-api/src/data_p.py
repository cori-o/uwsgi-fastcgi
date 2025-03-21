import pandas as pd 
import numpy as np 
import hashlib
import os 

class DataProcessor:    
    def cleanse_text(self, text):
        '''
        다중 줄바꿈 제거 및 특수 문자 중복 제거
        '''
        import re 
        text = re.sub(r'(\n\s*)+\n+', '\n\n', text)
        text = re.sub(r"\·{1,}", " ", text)
        text = re.sub(r"\.{1,}", ".", text)
        return text

    def chunk_text(self, text, max_length=500, overlap=250):
        """
        벡터 임베딩 전, 텍스트 길이가 500자를 넘어가면 겹치는 부분(overlap)을 포함하여 분할.
        - 이전 청크의 마지막 250자 + 새로운 250자로 구성
        - 첫 번째 청크는 그대로 유지
        - (text_chunk, chunk_no) 리스트 반환
        """
        if len(text) <= max_length:
            return [(text, 1)]
        
        chunks = []
        chunk_no = 1
        chunks.append((text[:max_length], chunk_no))
        chunk_no += 1

        for i in range(max_length - overlap, len(text), max_length - overlap):
            chunk = text[i:i+max_length]
            chunks.append((chunk, chunk_no))
            chunk_no += 1
        return chunks

    def check_l2_threshold(self, txt, threshold, value):
        threshold_txt = '' 
        print(f'Euclidean Distance: {value}, Threshold: {threshold}')
        if value > threshold:
            threshold_txt = '모르는 정보입니다.'
        else:
            threshold_txt = txt 
        return threshold_txt

    def hash_text(self, text, hash_type):
        if hash_type == 'blake':
            hashed_text = hashlib.blake2b(text.encode()).hexdigest() 
        elif hash_type == 'sha256':
            hashed_text = hashlib.sha256(text.encode()).hexdigest()
        return hashed_text

    def cohere_rerank(self, data):
        pass
