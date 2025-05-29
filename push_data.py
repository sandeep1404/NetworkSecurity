import os 
import sys
import json
from dotenv import load_dotenv
import certifi
import pandas as pd
import pymongo
import numpy as np
from networksecurity.exception.exception import NetworkSecurityException    
from networksecurity.logging.logger import logging


load_dotenv()

MANGO_DB_URL = os.getenv('MANGO_DB_URL')    
print(f"Using MANGO_DB_URL: {MANGO_DB_URL}")
ca= certifi.where() ## setuo secure https connection to mongodb atlas by trusted certificates.(ca is cenrtificate authority)


class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records= list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_to_mongodb(self,records,database,collection):
        try:
            self.database= database
            self.collection= collection
            self.records= records
            self.mango_client = pymongo.MongoClient(MANGO_DB_URL)

            self.database = self.mango_client[self.database]
            self.collection = self.database[self.collection]

            self.collection.insert_many(self.records)

            return(len(self.records))

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__ == '__main__':
    FILE_PATH = '/home/sandeep/Desktop/MLops/NetworkSecurity/Network_data/phisingData.csv'
    DATABASE = 'TECHIOT'
    COLLECTION = 'NetworkData'
    networkobj = NetworkDataExtract()
    records= networkobj.csv_to_json_convertor(FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_to_mongodb(records, DATABASE, COLLECTION)
    print(f"Number of records inserted: {no_of_records}")







