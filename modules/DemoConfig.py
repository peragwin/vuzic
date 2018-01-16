import graphene


class Config:

    def makeResolver(self, attr):
        return lambda root, args, context, info: getattr(self, attr)

    class Directions(graphene.Enum):
        IN = 1
        OUT = -1

    def __init__(self,
                 global_brightness: int,
                 direction: int,
                 amp_gain: float,
                 diff_gain: float,
                 amp_offset: int,
                 brightness: int,
                 space_period: int,
                 channel_sync: float):

        self.global_brightness = global_brightness
        self.direction = direction
        self.amp_gain = amp_gain
        self.diff_gain = diff_gain
        self.amp_offset = amp_offset
        self.brightness = brightness
        self.space_period = space_period
        self.channel_sync = channel_sync

        # Since this class is basically a singleton, I think it's okay to define the Query
        # like this  otherwise I would use the "context" part of the resolver methods to
        # pass an instance of the config
        class _Config(graphene.AbstractType):

            global_brightness = graphene.Int(
                resolver=self.makeResolver('global_brightness'))

            direction = self.Directions(
                resolver=self.makeResolver('direction'))

            amp_gain = graphene.Int(
                resolver=self.makeResolver('amp_gain'))

            diff_gain = graphene.Float(
                resolver=self.makeResolver('diff_gain'))

            amp_offset = graphene.Int(
                resolver=self.makeResolver('amp_offset'))

            brightness = graphene.Int(
                resolver=self.makeResolver('brightness'))

            space_period = graphene.Int(
                resolver=self.makeResolver('space_period'))

            channel_sync = graphene.Float(
                resolver=self.makeResolver('channel_sync'))

        class Query(graphene.ObjectType, _Config):
            pass

        self.Query = Query

        class QueryInput(graphene.InputObjectType, _Config):
            pass

        class Update(graphene.Mutation, _Config):
            class Input:
                input = graphene.Argument(QueryInput)

            # seems simple enough to just inherit from the AbstractType for now
            # config = graphene.Field(lambda: Query)

            @staticmethod
            def mutate(root, args, context, info):
                ret = {}
                input = args.get('input')
                for f in input.keys():
                    v = input.get(f)
                    if v:
                        setattr(self, f, v)
                    ret[f] = getattr(self, f)

                return Update(**ret)

        class Mutations(graphene.ObjectType):
            update = Update.Field()

        self.Mutations = Mutations


if __name__ == '__main__':
    from graphene.test import Client
    from pprint import pprint

    config = Config(10, Config.Directions.OUT, 20, 1e-3, 30, 40, 50, 2e-3)

    schema = graphene.Schema(query=config.Query, mutation=config.Mutations)
    client = Client(schema)

    # Test Query

    ex = client.execute("{ globalBrightness }")
    assert ex.get('errors', None) is None, ex['errors']
    pprint(ex['data'])
    assert ex['data'] == { 'globalBrightness': 10 }

    # Test Update

    ex = client.execute("mutation { update(input: { globalBrightness:100 }) { globalBrightness } }")
    assert ex.get('errors', None) is None, ex['errors']
    pprint(ex['data'])
    assert ex['data'] == { 'update': { 'globalBrightness': 100 } }

    assert config.global_brightness == 100, config.global_brightness