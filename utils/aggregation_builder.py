class AggregationBuilder:
    def __init__(self):
        self.__aggregations = []

        self.convert_ops = {
            'tostring': 'string',
            'toobjectid': 'objectId'
        }

    # tos
    def convert_field(self, converted_field_name, ori_field_name, to):
        return {
            converted_field_name: {
                '$convert': {
                    'input': '$' + ori_field_name,
                    'to': self.convert_ops[to]
                }
            }
        }

    def match(self, query):
        self.__aggregations.append({
            '$match': query
        })

        return self
    
    def lookup(self, from_table, local_field, foreign_field, rename):
        self.__aggregations.append({
            '$lookup': {
                'from': from_table,
                'localField': local_field,
                'foreignField': foreign_field,
                'as': rename
            }
        })

        return self
    
    def project(self, field_list, converts, keep=True):
        projects = {
            f: 1 if keep else 0 
            for f in field_list
        }

        if len(converts) > 0:
            for c in converts:
                projects.update(c)

        self.__aggregations.append({
            '$project': projects
        })

        return self
    
    def groupby(self, field, search_params):
        self.__aggregations.append({
            '$group': {
                '_id': '$' + field,
                **search_params
            }
        })

        return self
    
    def sort(self, fields):
        self.__aggregations.append({
            '$sort': fields
        })

        return self
    
    def unwind(self, field):
        self.__aggregations.append({
            '$unwind': '$' + field
        })

        return self
    
    def result(self):
        return self.__aggregations