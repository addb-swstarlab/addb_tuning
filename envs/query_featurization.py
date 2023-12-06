import numpy as np
import pandas as pd
import sqlparse
import re
class QueryFeatureGenerator:
    def __init__(self, file_path, positional_encoding=False, period_factor=False, amplitude_factor=0.2):
        self.file_path = file_path
        
        # positional encoding   #################
        self.apply_pe = positional_encoding
        self.period_factor = period_factor
        self.amplitude_factor = amplitude_factor
        # positional encoding   #################
        
        self.query_feature_list = ['joins', 'filters', 'aggregation', 'sorting', 'subqueries', 'grouping']
        self.table_list = ['customer', 'nation', 'part', 'region', 'lineitem', 'orders', 'partsupp', 'supplier']
        self.entire_sql = self.load_sql_file()  
        self.splited_sql = self.split_sql()
        self.sequence_feature = self.get_query_vector()
        self.workload_feature = self.sequence_to_workload() # summation each column of self.sequence_feature
        
    def load_sql_file(self):
        with open(self.file_path, 'r') as file:
            sql_query = file.read().lower()
        return sql_query
    
    def split_sql(self):
        splited_sql = sqlparse.split(self.entire_sql) # [query1, query2, ...]   type(query1) --> str
        return splited_sql
    
    def sequence_to_workload(self):
        column_sums = self.sequence_feature.sum()
        sum_df = pd.DataFrame([column_sums])   
        return sum_df

    def get_query_vector(self):

        self.query_sequence_feature_df = pd.DataFrame(columns=(self.query_feature_list + self.table_list)) 
        
        for i in range(len(self.splited_sql)):
            sql_query = self.splited_sql[i]
            
            '''get table vector for splited sql'''
            table_vector = self.get_table_vector(sql_query)
            
            
            '''get query vector for splited sql'''            
            
            joins, filters, aggregation, sorting, subqueries, grouping = 0, 0, 0, 0, 0, 0
            
            # Filters
            filters += self.count_filters(sql_query)            
            # Joins
            joins += len(re.findall(r'join', sql_query, re.IGNORECASE))           
            # Aggregation
            aggregation += len(re.findall(r'\b(sum|count|avg|min|max)\(', sql_query, re.IGNORECASE))             
            # Subqueries
            select_cnt = len(re.findall(r'\bselect\b', sql_query, re.IGNORECASE))
            if select_cnt > 1: subqueries += 1            
            # Grouping
            grouping_cnt = re.findall(r'group by', sql_query, re.IGNORECASE)
            if len(grouping_cnt) >= 1: grouping += 1
            # Sorting
            sorting_cnt = re.findall(r'order by', sql_query, re.IGNORECASE)
            if len(sorting_cnt) >= 1: sorting += 1
                           
            sql_vector = pd.DataFrame([{
                'joins': joins,
                'filters': filters,
                'aggregation': aggregation,
                'sorting': sorting,
                'subqueries': subqueries,
                'grouping': grouping
            }])
                
            query_vector = pd.concat([sql_vector, table_vector], axis=1)
            self.query_sequence_feature_df = pd.concat([self.query_sequence_feature_df, query_vector], axis=0, ignore_index=True)

        self.query_sequence_feature_df.reset_index(drop=True)
        
        if self.apply_pe:   # when positional encoding is True
            encoded_query_vector = self.add_positional_encoding(self.query_sequence_feature_df)        
            return encoded_query_vector
        
        return self.query_sequence_feature_df
        

    def get_table_vector(self, query):        
        table_vector = {}
        formatted_entire_query = sqlparse.format(query, reindent=True, keyword_case='upper')
        
        for i in range(len(self.table_list)):      
            key = self.table_list[i]
            
            pattern = re.compile(rf'\b {key}\b|{key},|{key}\n', re.IGNORECASE)      
            find_table = pattern.findall(formatted_entire_query)
            
            if len(find_table) >= 1:
                table_vector[key] = 1
            else:
                table_vector[key] = 0
                
        table_vector_df = pd.DataFrame([table_vector])
        
        return table_vector_df
    
        
    def count_filters(self, sql_query):
        formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
        
        where = re.findall(r'\bWHERE\b ', formatted_sql)
        and_or = re.findall(r'\bAND\b |\bOR\b ', formatted_sql)
        having = re.findall(r'\bHAVING\b ', formatted_sql)
        case = re.findall(r'\bCASE\b', formatted_sql)

        filters_num = len(where) + len(and_or) + len(having) + len(case)
        
        return filters_num


    def add_positional_encoding(self, df): 
        n, m = df.shape # n : row(num of sequence)  |    m : column(number of feature)
        af = self.amplitude_factor   # value range of positional eoncoding is [-af, +af] / if af=1 --> [-1, 1]
        pf = self.period_factor
        
        if pf == False:        
            pf = (10000/512)*m  # positional encoding in transformer :  period_factor  is 10000   |   feature dimension of embedding vector is 512

        # Positional Encoding
        # PE(pos, 2i) = af*sin(pos / pf^(2i/d_model))
        # PE(pos, 2i+1) = af*cos(pos / pf^(2i/d_model))
        # pos is sequence of query, i is index of feature dimension
        pos_encoding = np.array([
            [pos / np.power(pf, 2 * (i // 2) / m) for i in range(m)]
            for pos in range(n)
        ])

        pos_encoding[:, 0::2] = af*(np.sin(pos_encoding[:, 0::2]))  # 2i    --> sin
        pos_encoding[:, 1::2] = af*(np.cos(pos_encoding[:, 1::2])) # 2i+1   --> cos
        
        self.pos_df = pd.DataFrame(pos_encoding, columns=df.columns)        
#         encoded_df = df.add(self.pos_df) # add positional encoding 
        encoded_df = df.multiply(self.pos_df) # multiply positional encoding 
        encoded_df = encoded_df.add(df)
        
        return encoded_df

# file_path = 'file_path'
# q_feature_vector = query_featurization(file_path, positional_encoding=True, amplitude_factor=0.2)
# print(q_feature_vector.sequence_feature)
# print(q_feature_vector.pos_df)
# print(q_feature_vector.workload_feature)
