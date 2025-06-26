from motor.motor_asyncio import AsyncIOMotorClient
from motor.core import AgnosticClient
from pymongo import ReturnDocument, UpdateOne, InsertOne, UpdateMany, DeleteMany, ReplaceOne, WriteConcern
\
from utils.singleton_wrapper import Singleton

from hamunafs.utils.redisutil import XRedisAsync

import asyncio
import copy
import traceback

class AsyncMongoEngine(Singleton):
    def __init__(self, host=None, port=None, username=None, password=None):
        # self.host = host
        # self.port = port
        # self.username = username
        # self.password = password

        self.mongo_url = 'mongodb://'
        if username is not None and password is not None:
            self.mongo_url += '{}:{}@'.format(username, password)
        
        self.mongo_url += '{}:{}'.format(host, port)

    def connect(self):
        self.client = AsyncIOMotorClient(self.mongo_url, serverSelectionTimeoutMS=1000 * 3)

    async def connect_async(self, loop=None):
        AgnosticClient.get_io_loop = asyncio.get_running_loop
        self.client = AsyncIOMotorClient(self.mongo_url, serverSelectionTimeoutMS=1000 * 3)

    async def start_transaction(self):
        return await self.client.start_session(causal_consistency=True)

    def get_db(self, cluster):
        return self.client[cluster]

    async def create_index(self, db, key, index, unique=False, background=True):
        try:
            if isinstance(db, str):
                db = self.get_db(db)
            
            await self.create_collection(db, key)

            await db[key].create_index(index, background=background, unique=unique)
            return True
        except Exception as e:
            return False

    async def create_collection(self, db, key, session=None):
        try:
            await db.create_collection(key)
        except:
            pass

    async def add_one(self, db, key, _dict, exist_query={}, session=None):
        try:
            if isinstance(db, str):
                db = self.get_db(db)
            
            await self.create_collection(db, key, session)

            if len(exist_query) > 0:
                count = await db[key].count_documents(exist_query)
                if count > 0:
                    return False, '信息重复'

            result = await db[key].insert_one(
                copy.deepcopy(_dict), session=session)
            return True, result.inserted_id
        except Exception as e:
            return False, str(e)

    async def add_many(self, db, key, list, session=None):
        try:
            if isinstance(db, str):
                db = self.get_db(db)
            
            await self.create_collection(db, key, session=session)

            result = await db[key].insert_many(
                list, session=session, ordered=False)
            return True, result.inserted_ids
        except Exception as e:
            return False, str(e)

    async def query_one(self, db, key, query, project=None):
        if isinstance(db, str):
            db = self.get_db(db)
        item = await db[key].find_one(query, project)
        if item is not None:
            item['id'] = item['_id'].__str__()
            del item['_id']

        return item

    async def query(self, db, key, query, skip=None, limit=None):
        if isinstance(db, str):
            db = self.get_db(db)
        data = await self.query_and_sort(db, key, query, skip=skip, limit=limit)
        return data

    async def query_and_sort(self, db, key, query, project=None, sort_key=None, sort_order=1, skip=None, limit=None, need_count=False):
        if isinstance(db, str):
            db = self.get_db(db)
        base = db[key].find(query, project)

        if sort_key is not None:
            base = base.sort(sort_key, sort_order)

        if limit is not None:
            if limit > 0:
                if isinstance(limit, int):
                    if skip is None:
                        skip = 0
                    base = base.clone().limit(int(limit)).skip(int(skip))

        output = [self.__replace_id(o) async for o in base]

        if need_count:
            if len(query) > 0:
                count = await db[key].count_documents(query)
            else:
                count = await db[key].estimated_document_count()
            return output, count

        return output

    async def count(self, db, key, query, expired=7200, refresh_cache=False, cachable=False, distinct=None):
        query_keys = list(query.keys())
        query_keys.sort()
        extra_keys = ''.join([str(query[q]) for q in query_keys])
        cache_key = '{}_{}'.format(key, extra_keys)

        item = await self.get_cache(
            cache_key, return_obj=False) if (not refresh_cache and cachable) else None

        if item is None:
            if cachable:
                async with await self.cache_manager.lock(cache_key, ttl=30):
                    item = await self.get_cache(
                        cache_key) if (not refresh_cache and cachable) else None
                    if item is None:
                        if isinstance(db, str):
                            db = self.get_db(db)

                        if len(query) > 0 or distinct is not None:
                            if distinct is not None:
                                item = await db[key].distinct(distinct)
                                item = await item.count_document(query)
                            else:
                                item = await db[key].count_documents(query)
                        else:
                            item = await db[key].estimated_document_count()
                        if item is not None and cachable:
                            await self.set_cache(
                                cache_key, item, expired=expired)
            else:
                item = await self.get_cache(
                    cache_key) if (not refresh_cache and cachable) else None
                if item is None:
                    if isinstance(db, str):
                        db = self.get_db(db)

                    if len(query) > 0 or distinct is not None:
                        if distinct is not None:
                            item = db[key].aggregate([
                                {'$match':query},
                                {'$group':{'_id':"${}".format(distinct)}},
                                {'$group':{'_id':None, 'count':{'$sum':1}}}
                            ], allowDiskUse=True)
                            item = await item.to_list(length=None)
                            if len(item) == 0:
                                item = 0
                            else:
                                item = item[0]['count']
                        else:
                            item = await db[key].count_documents(query)
                    else:
                        item = await db[key].estimated_document_count()
                    if item is not None and cachable:
                        await self.set_cache(
                            cache_key, item, expired=expired)
        return 0 if item is None else int(item)

    async def update_one(self, db, key, query, update_data, return_entity=False, session=None, upsert=False):
        if isinstance(db, str):
            db = self.get_db(db)

        if upsert:
            await self.create_collection(db, key, session=session)

        results = await db[key].find_one_and_update(
            query, update_data, session=session, upsert=upsert, return_document=ReturnDocument.AFTER)
        if return_entity and results is not None:
            results['id'] = results['_id'].__str__()
            del results['_id']
            return results is not None, results
        else:
            if upsert and results is None:
                return True, None
            return results is not None, None

    async def update(self, db, key, query, update_data, session=None, upsert=False, return_entity=False):
        if isinstance(db, str):
            db = self.get_db(db)
        
        if upsert:
            await self.create_collection(db, key, session=session)

        results = await db[key].update_many(query, update_data, session=session, upsert=upsert)

        return results.acknowledged

    def __replace_id(self, item):
        if '_id' in item:
            item['id'] = item['_id'].__str__()
            del item['_id']
        return item

    async def aggregate(self, cache_key, db, key, query, refresh_cache=False, need_count=False, count_query={}, expired=7200):
        try:
            if isinstance(db, str):
                db = self.get_db(db)
            results = db[key].aggregate(query, allowDiskUse=True)
            results = await results.to_list(length=None)

            if need_count:
                if len(count_query) > 0:
                    count = await db[key].count_documents(count_query)
                else:
                    count = await db[key].estimated_document_count()
                return (results, count)
            return results
        except Exception as e:
            traceback.print_exc()
            return None

    async def remove(self, db, key, query, session=None):
        if isinstance(db, str):
            db = self.get_db(db)
        
        result = await db[key].delete_one(query, session=session)
        return result.deleted_count >= 0

    async def remove_on_query(self, db, key, query, session=None):
        if isinstance(db, str):
            db = self.get_db(db)
        
        result = await db[key].delete_many(query, session=session)
        return result.deleted_count >= 0

    async def remove_collection(self, db, collection_name, session=None):
        if isinstance(db, str):
            db = self.get_db(db)
        return await db.drop_collection(collection_name, session=session)

    async def merge(self, db, source, target):
        # db.<SOURCE_COLLECTION>.aggregate([ { $match: {} }, { $out: "<TARGET_COLLECTION>" } ])
        if isinstance(db, str):
            db = self.get_db(db)

        await self.create_collection(db, target)

        pipeline = [{'$merge': {'into': target, 'on': '_id', 'whenMatched': 'replace', 'whenNotMatched': 'insert'}}]

        ops = db.get_collection(source).aggregate(pipeline)

        async for op in ops:
            print(op)

        return ops

    async def bulk_write(self, db, key, requests):
        if isinstance(db, str):
            db = self.get_db(db)
        
        await self.create_collection(db, key)

        try:
            result = await db[key].bulk_write(requests)
            return len(result.bulk_api_result['writeErrors']) == 0
        except:
            return False

    async def get_collection_list(self, db):
        if isinstance(db, str):
            db = self.get_db(db)
        coll_names = await db.list_collection_names(session=None)
        return coll_names
