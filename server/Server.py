import asyncio
import threading

from aiohttp import web
from graphene import Schema

class Server:

	def __init__(self,
				 host: str = 'localhost',
				 port: int = 8080,
				 schema: Schema = None):

		self.schema = schema
		self.host = host
		self.port = port

		self.loop = loop = asyncio.get_event_loop()
		app = web.Application(loop=loop)

		app.router.add_get('/', self.index)
		app.router.add_route('*', '/graphql', self.graphql)

		self.app = app

	def start(self):
		t = threading.Thread(target=self.loop.run_until_complete(self._start()))
		t.start()

	async def _start(self):
		self.server = await self.loop.create_server(
			self.app.make_handler(), self.host, self.port)
		print("started server")
		await self.server.wait_closed()

	def stop(self):
		self.server.close()
		print('stopping server')

	async def index(self, request):
		print("GET => '/'")
		return web.Response(text="test index")

	async def graphql(self, request):
		print("POST => '/graphql'", request)
		query = await request.text()
		resp = self.schema.execute(query)
		return web.json_response(resp)


if __name__ == '__main__':
	S = Server()
	try:
		S.start()
	except KeyboardInterrupt:
		print('interrupt')

	S.stop()
