from bson import ObjectId

from settings import mongodb


class EvidenceManager:
    @staticmethod
    def get(user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        res = mongodb.memm_evid.find({'user_id': user_id})
        evidences = {
            str(doc['user_id']):
                [
                    doc['dimension'],
                    [
                        [
                            [int(obs_state[0]), obs_state[1]] for obs_state in seq
                        ] for seq in doc['evidences']
                    ]
                ] for doc in res
        }
        return evidences
