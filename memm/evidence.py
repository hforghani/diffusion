from bson import ObjectId

from settings import mongodb


class EvidenceManager:
    @staticmethod
    def get(user_id):
        if not isinstance(user_id, ObjectId):
            user_id = ObjectId(user_id)
        doc = mongodb.memm_evid.find_one({'user_id': user_id})
        if doc is None:
            return None
        else:
            evidences = [
                doc['dimension'],
                [
                    [
                        [int(obs_state[0]), obs_state[1]] for obs_state in seq
                    ] for seq in doc['evidences']
                ]
            ]
            return evidences
