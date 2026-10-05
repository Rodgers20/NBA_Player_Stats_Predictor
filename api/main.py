"""REST entry point. Startup and local data reads never refresh paid odds."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import props, games, players, hitrates

app = FastAPI(title='Basketball Props API', version='1.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('API_CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(','),
    allow_credentials=False, allow_methods=['GET', 'POST', 'PATCH', 'DELETE'], allow_headers=['*'],
)
for router in (props.router, games.router, players.router, hitrates.router):
    app.include_router(router, prefix='/api')


@app.get('/api/health')
def health():
    return {'status': 'ok'}


from api.routes import journal
app.include_router(journal.router, prefix='/api')

if os.getenv('SERVE_FRONTEND') == '1':
    from pathlib import Path
    from starlette.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException as StarletteHTTPException

    class ExportedFrontend(StaticFiles):
        async def get_response(self, path, scope):
            if path == 'api' or path.startswith('api/'):
                raise StarletteHTTPException(404)
            try:
                response = await super().get_response(path, scope)
                if response.status_code != 404 or Path(path).suffix:
                    return response
            except StarletteHTTPException as exc:
                if exc.status_code != 404 or Path(path).suffix:
                    raise
            return await super().get_response(path + '.html', scope)

    frontend_dir = Path(os.getenv('FRONTEND_DIR', str(Path(__file__).resolve().parents[1] / 'frontend' / 'out')))
    app.mount('/', ExportedFrontend(directory=frontend_dir, html=True), name='frontend')
